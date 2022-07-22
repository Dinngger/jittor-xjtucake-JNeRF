export PYTHONPATH=$PYTHONPATH:./python
unset LD_LIBRARY_PATH
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_comp.py --task B_test
# "train","test","B_test","render","val_all","gui"
