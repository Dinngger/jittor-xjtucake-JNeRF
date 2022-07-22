from cmath import pi
from math import fmod, cos, sin
import os
import time
import jittor as jt
from PIL import Image
import numpy as np
from tqdm import tqdm
from jnerf.ops.code_ops import *
from jnerf.dataset.dataset import jt_srgb_to_linear, jt_linear_to_srgb
from jnerf.utils.config import get_cfg, save_cfg
from jnerf.utils.registry import build_from_cfg,NETWORKS,SCHEDULERS,DATASETS,OPTIMS,SAMPLERS,LOSSES
from jnerf.models.losses.mse_loss import img2mse, mse2psnr
from jnerf.dataset import camera_path
import cv2
import dearpygui.dearpygui as dpg

class Runner():
    def __init__(self):
        self.cfg = get_cfg()
        if self.cfg.fp16 and jt.flags.cuda_archs[0] < 70:
            print("Warning: Sm arch is lower than sm_70, fp16 is not supported. Automatically use fp32 instead.")
            self.cfg.fp16 = False
        if not os.path.exists(self.cfg.log_dir):
            os.makedirs(self.cfg.log_dir)
        self.exp_name           = self.cfg.exp_name
        self.dataset            = {}
        self.cfg.dataset_obj    = None
        self.dataset["train"]   = None
        self.dataset["val"]     = None
        self.dataset["test"]    = None
    
    def build_model(self):
        self.model              = build_from_cfg(self.cfg.model, NETWORKS)
        self.cfg.model_obj      = self.model
        self.sampler            = build_from_cfg(self.cfg.sampler, SAMPLERS)
        self.cfg.sampler_obj    = self.sampler
        self.optimizer          = build_from_cfg(self.cfg.optim, OPTIMS, params=self.model.parameters())
        self.optimizer          = build_from_cfg(self.cfg.expdecay, OPTIMS, nested_optimizer=self.optimizer)
        self.ema_optimizer      = build_from_cfg(self.cfg.ema, OPTIMS, params=self.model.parameters())
        self.loss_func          = build_from_cfg(self.cfg.loss, LOSSES)
        self.background_color   = self.cfg.background_color
        self.tot_train_steps    = self.cfg.tot_train_steps
        self.n_rays_per_batch   = self.cfg.n_rays_per_batch
        self.using_fp16         = self.cfg.fp16
        self.save_path          = os.path.join(self.cfg.log_dir, self.exp_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.cfg.ckpt_path and self.cfg.ckpt_path is not None:
            self.ckpt_path = self.cfg.ckpt_path
        else:
            self.ckpt_path = os.path.join(self.save_path, "params.pkl")
        if self.cfg.load_ckpt:
            self.load_ckpt(self.ckpt_path)
        else:
            self.start=0

        self.cfg.m_training_step = 0
        self.val_freq = 4096

    def train(self):
        if self.dataset["train"] is None:
            self.dataset["train"]   = build_from_cfg(self.cfg.dataset.train, DATASETS)
            if self.cfg.dataset_obj is None:
                self.cfg.dataset_obj = self.dataset["train"]
                self.build_model()
            self.image_resolutions = self.dataset["train"].resolution
            self.W = self.image_resolutions[0]
            self.H = self.image_resolutions[1]
        old_training_step = self.start
        tqdm_last_update = time.monotonic()
        with tqdm(desc="Training", total=self.tot_train_steps, unit="step", initial=self.start) as t:
            for i in range(self.start, self.tot_train_steps):
                self.cfg.m_training_step = i
                img_ids, rays_o, rays_d, rgb_target = next(self.dataset["train"])
                training_background_color = jt.random([rgb_target.shape[0],3]).stop_grad()

                rgb_target = (rgb_target[..., :3] * rgb_target[..., 3:] + training_background_color * (1 - rgb_target[..., 3:])).detach()

                pos, dir = self.sampler.sample(img_ids, rays_o, rays_d, is_training=True)
                network_outputs = self.model(pos, dir)
                rgb = self.sampler.rays2rgb(network_outputs, training_background_color)

                loss = self.loss_func(rgb, rgb_target)
                self.optimizer.step(loss)
                self.ema_optimizer.ema_step()
                if self.using_fp16:
                    self.model.set_fp16()

                if i < old_training_step or old_training_step == 0:
                    old_training_step = 0
                    t.reset()
                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    t.update(i - old_training_step)
                    t.set_postfix(psnr=mse2psnr(loss.mean()/5).item())
                    old_training_step = i
                    tqdm_last_update = now
        self.save_ckpt(os.path.join(self.save_path, "params.pkl"))
        # self.val_all()
        # self.test()

    def test(self, load_ckpt=False, B_test=False):
        if self.dataset["test"] is None:
            if B_test:
                self.dataset["test"] = build_from_cfg(self.cfg.dataset.B_test, DATASETS)
            else:
                self.dataset["test"] = build_from_cfg(self.cfg.dataset.test, DATASETS)
            if self.cfg.dataset_obj is None:
                self.cfg.dataset_obj = self.dataset["test"]
                self.build_model()
            self.image_resolutions = self.dataset["test"].resolution
            self.W = self.image_resolutions[0]
            self.H = self.image_resolutions[1]
        if load_ckpt:
            assert os.path.exists(self.ckpt_path), "ckpt file does not exist: "+self.ckpt_path
            self.load_ckpt(self.ckpt_path)
        if not os.path.exists('./result'):
            os.makedirs('./result')
        mse_list=self.render_test(save_path='./result')
        if self.dataset["test"].have_img:
            tot_psnr=0
            for mse in mse_list:
                tot_psnr += mse2psnr(mse)
            print("TOTAL TEST PSNR===={}".format(tot_psnr/len(mse_list)))

    def render(self, load_ckpt=True, save_path=None):
        if load_ckpt:
            assert os.path.exists(self.ckpt_path), "ckpt file does not exist: "+self.ckpt_path
            self.load_ckpt(self.ckpt_path)
        if save_path is None or save_path=="":
            save_path = os.path.join(self.save_path, "demo.mp4")
        else:
            assert save_path.endswith(".mp4"), "suffix of save_path need to be .mp4"
        print("rendering video with specified camera path")
        fps = 28
        W, H = self.image_resolutions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videowriter = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
        cam_path = camera_path.path_spherical()

        for pose in tqdm(cam_path):
            img = self.render_img_with_pose(pose)
            img = (img*255+0.5).clip(0, 255).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            videowriter.write(img)
        videowriter.release()
        
    def save_ckpt(self, path):
        jt.save({
            'global_step': self.cfg.m_training_step,
            'model': self.model.state_dict(),
            'sampler': self.sampler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'nested_optimizer': self.optimizer._nested_optimizer.state_dict(),
            'ema_optimizer': self.ema_optimizer.state_dict(),
        }, path)

    def load_ckpt(self, path):
        print("Loading ckpt from:",path)
        ckpt = jt.load(path)
        self.start = ckpt['global_step']
        self.model.load_state_dict(ckpt['model'])
        if self.using_fp16:
            self.model.set_fp16()
        self.sampler.load_state_dict(ckpt['sampler'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        nested=ckpt['nested_optimizer']['defaults']['param_groups'][0]
        for pg in self.optimizer._nested_optimizer.param_groups:
            for i in range(len(pg["params"])):
                pg["values"][i]=jt.array(nested["values"][i])
                pg["m"][i]=jt.array(nested["m"][i])
        ema=ckpt['ema_optimizer']['defaults']['param_groups'][0]
        for pg in self.ema_optimizer.param_groups:
            for i in range(len(pg["params"])):
                pg["values"][i]=jt.array(ema["values"][i])
        self.ema_optimizer.steps=ckpt['ema_optimizer']['defaults']['steps']
        
    def val_img(self, iter):
        with jt.no_grad():
            img, img_tar = self.render_img(dataset_mode="val")
            self.save_img(self.save_path+f"/img{iter}.png", img)
            self.save_img(self.save_path+f"/target{iter}.png", img_tar)
            return img2mse(
                jt.array(img), 
                jt.array(img_tar)).item()

    def val_all(self):
        if self.dataset["val"] is None:
            if self.cfg.dataset.val:
                self.dataset["val"] = build_from_cfg(self.cfg.dataset.val, DATASETS)
                if self.cfg.dataset_obj is None:
                    self.cfg.dataset_obj = self.dataset["val"]
                    self.build_model()
                    self.image_resolutions = self.dataset["val"].resolution
                    self.W = self.image_resolutions[0]
                    self.H = self.image_resolutions[1]
                    assert os.path.exists(self.ckpt_path), "ckpt file does not exist: "+self.ckpt_path
                    self.load_ckpt(self.ckpt_path)
            else:
                print('no val dateset!')
                return
        totpsnr = 0
        minpsnr = 1000
        maxpsnr = 0
        min_img = None
        min_tar = None
        with tqdm(range(0,self.dataset["val"].n_images,1), unit="images", desc="rendering val images") as t:
            for i in t:
                with jt.no_grad():
                    img, img_tar = self.render_img(dataset_mode="val")
                    mse = img2mse(jt.array(img), jt.array(img_tar)).item()
                    psnr = mse2psnr(mse)
                    totpsnr += psnr
                    if psnr < minpsnr:
                        min_img = img
                        min_tar = img_tar
                    minpsnr = psnr if psnr<minpsnr else minpsnr
                    maxpsnr = psnr if psnr>maxpsnr else maxpsnr
        psnr = totpsnr/(self.dataset["val"].n_images or 1)
        print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}]")
        self.save_img(self.save_path+f"/min_img.png", min_img)
        self.save_img(self.save_path+f"/min_tar.png", min_tar)

    def render_test(self, save_img=True, save_path=None):
        if save_path is None:
            save_path = self.save_path
        mse_list = []
        print("rendering testset...")
        for img_i in tqdm(range(0,self.dataset["test"].n_images,1)):
            with jt.no_grad():
                imgs=[]
                for i in range(1):
                    simg, img_tar = self.render_img(dataset_mode="test", img_id=img_i)
                    imgs.append(simg)
                img = np.stack(imgs, axis=0).mean(0)
                if save_img:
                    self.save_img(save_path+f"/{self.exp_name}_r_{img_i}.png", img)
                    # if self.dataset["test"].have_img:
                    #     self.save_img(save_path+f"/{self.exp_name}_gt_{img_i}.png", img_tar)
                mse_list.append(img2mse(
                jt.array(img), 
                jt.array(img_tar)).item())
        return mse_list

    def save_img(self, path, img):
        if isinstance(img, np.ndarray):
            ndarr = (img*255+0.5).clip(0, 255).astype('uint8')
        elif isinstance(img, jt.Var):
            ndarr = (img*255+0.5).clamp(0, 255).uint8().numpy()
        im = Image.fromarray(ndarr)
        im.save(path)

    def render_img(self, dataset_mode="train", img_id=None):
        W, H = self.image_resolutions
        H = int(H)
        W = int(W)
        if img_id is None:
            img_id = np.random.randint(0, self.dataset[dataset_mode].n_images, [1])[0]
            img_ids = jt.zeros([H*W], 'int32')+img_id
        else:
            img_ids = jt.zeros([H*W], 'int32')+img_id
        rays_o_total, rays_d_total, rays_pix_total = self.dataset[dataset_mode].generate_rays_total_test(
            img_ids, W, H)
        rays_pix_total = rays_pix_total.unsqueeze(-1)
        pixel = 0
        imgs = np.empty([H*W+self.n_rays_per_batch, 3])
        for pixel in range(0, W*H, self.n_rays_per_batch):
            end = pixel+self.n_rays_per_batch
            rays_o = rays_o_total[pixel:end]
            rays_d = rays_d_total[pixel:end]
            if end > H*W:
                rays_o = jt.concat(
                    [rays_o, jt.ones([end-H*W]+rays_o.shape[1:], rays_o.dtype)], dim=0)
                rays_d = jt.concat(
                    [rays_d, jt.ones([end-H*W]+rays_d.shape[1:], rays_d.dtype)], dim=0)

            pos, dir = self.sampler.sample(img_ids, rays_o, rays_d)
            network_outputs = self.model(pos, dir)
            rgb = self.sampler.rays2rgb(network_outputs, inference=True)
            imgs[pixel:end] = rgb.numpy()
            jt.sync_all()
            jt.gc()
        imgs = imgs[:H*W].reshape(H, W, 3)
        imgs_tar=jt.array(self.dataset[dataset_mode].image_data[img_id]).reshape(H, W, 4)
        imgs_tar = imgs_tar[..., :3] * imgs_tar[..., 3:] + jt.array(self.background_color) * (1 - imgs_tar[..., 3:])
        imgs_tar = imgs_tar.detach().numpy()
        jt.sync_all()
        jt.gc()
        return imgs, imgs_tar

    def render_img_with_pose(self, pose, dataset_mode="train"):
        W, H = self.image_resolutions
        H = int(H)
        W = int(W)
        fake_img_ids = jt.zeros([H*W], 'int32')
        rays_o_total, rays_d_total = self.dataset[dataset_mode].generate_rays_with_pose(pose, W, H)
        img = np.empty([H*W+self.n_rays_per_batch, 3], dtype=np.float32)
        for pixel in range(0, W*H, self.n_rays_per_batch):
            end = pixel+self.n_rays_per_batch
            rays_o = rays_o_total[pixel:end]
            rays_d = rays_d_total[pixel:end]
            if end > H*W:
                rays_o = jt.concat(
                    [rays_o, jt.ones([end-H*W]+rays_o.shape[1:], rays_o.dtype)], dim=0)
                rays_d = jt.concat(
                    [rays_d, jt.ones([end-H*W]+rays_d.shape[1:], rays_d.dtype)], dim=0)
            pos, dir = self.sampler.sample(fake_img_ids, rays_o, rays_d)
            network_outputs = self.model(pos, dir)
            rgb = self.sampler.rays2rgb(network_outputs, inference=True)
            img[pixel:end] = rgb.numpy()
        img = img[:H*W].reshape(H, W, 3)
        jt.gc()
        return img

    def gui(self, load_ckpt=True):
        if self.dataset["test"] is None:
            self.dataset["test"] = build_from_cfg(self.cfg.dataset.test, DATASETS)
            if self.cfg.dataset_obj is None:
                self.cfg.dataset_obj = self.dataset["test"]
                self.build_model()
            self.image_resolutions = self.dataset["test"].resolution
            self.W = self.image_resolutions[0]
            self.H = self.image_resolutions[1]
        if load_ckpt:
            assert os.path.exists(self.ckpt_path), "ckpt file does not exist: "+self.ckpt_path
            self.load_ckpt(self.ckpt_path)
        
        m_camera = np.asarray([[1., 0., 0., 0.5],[0., -1., 0., 0.5],[0., 0., -1., 0.5]])
        m_scale = 8.0 * self.cfg.dataset_obj.aabb_scale
        m_up_dir = np.asarray([0., 1., 0.])
        view_pos = lambda: m_camera[:, 3]
        view_dir = lambda: m_camera[:, 2]
        m_camera[:, 3] -= m_scale * view_dir()
        image_pos = np.asarray([0., 0.])
        dpg.create_context()
        dpg.create_viewport(title='Jittor Graphics', width=1100, height=820)
        img = self.render_img_with_pose(jt.array(m_camera), "test")
        with dpg.texture_registry():
            dpg.add_raw_texture(width=800, height=800, default_value=img, format=dpg.mvFormat_Float_rgb, tag="render_tag")
        with dpg.window(tag="Primary Window"):
            dpg.add_image("render_tag")
        with dpg.window(label='JNeRF', pos=[820, 100]):
            dpg.add_text("Frame: %.2f ms (%.1f FPS); Mem: %s", tag="frame_text")
            with dpg.collapsing_header(label="Training", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Stop training")
                    dpg.add_checkbox(label="Train encoding")
        look_at = lambda: view_pos() + view_dir() * m_scale
        def rerender():
            nonlocal m_camera, img
            img[...] = self.render_img_with_pose(jt.array(m_camera), "test")
        def mouse_wheel_handler(m, delta):
            nonlocal image_pos, look_at, m_scale, m_camera
            scale_factor = pow(1.1, -delta)
            image_pos = (image_pos - m) / scale_factor + m
            prev_look_at = look_at()
            scale = m_scale * scale_factor
            m_camera[:, 3] = (view_pos() - prev_look_at) * (scale / m_scale) + prev_look_at
            m_scale = scale
            rerender()
        def set_look_at(pos):
            nonlocal m_camera, look_at
            m_camera[:, 3] += pos - look_at()
        def angleAxis(t, k):
            x, y, z = list(k)
            c = cos(t)
            s = sin(t)
            v = 1.0 - c
            return np.asarray([[x*x*v+c, x*y*v-z*s, x*z*v+y*s],
                               [x*y*v+z*s, y*y*v+c, y*z*v-x*s],
                               [x*z*v-y*s, y*z*v+x*s, z*z*v+c]])
        def mouse_drag_handler(button, rel_origin):
            nonlocal m_up_dir, m_camera, image_pos, look_at
            rel = np.asarray([-rel_origin[1], -rel_origin[2]], dtype=np.float)
            if (rel == np.array([0., 0.])).all():
                return
            up = m_up_dir
            side = m_camera[:, 0]
            is_left_held = button == 33
            is_right_held = button == dpg.mvMouseButton_Right
            is_middle_held = button == dpg.mvMouseButton_Middle
            if is_left_held:
                rot_sensitivity = 1.0 / 360.
                rot = angleAxis(-rel[0] * 2 * pi * rot_sensitivity, up) @ \
                      angleAxis(-rel[1] * 2 * pi * rot_sensitivity, side)
                image_pos += rel
                old_loot_at = look_at()
                set_look_at(np.asarray([0.0, 0.0, 0.0]))
                m_camera = rot @ m_camera
                set_look_at(old_loot_at)
            if is_right_held:
                rot = angleAxis(-rel[0] * 2 * pi, up) @ angleAxis(-rel[1] * 2 * pi, side)
            if is_middle_held:
                rel_ = np.asarray([-rel[0], -rel[1], 0.0])
                m_camera[:, 3] += m_camera[:, :3] @ rel_
            rerender()
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=mouse_wheel_handler)
            dpg.add_mouse_drag_handler(callback=mouse_drag_handler)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)
        dpg.start_dearpygui()
        dpg.destroy_context()
