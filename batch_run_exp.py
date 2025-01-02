#!/bin/sh
import argparse
import os
import libtmux
import time
import random
import psutil
import getpass
import pynvml
from os.path import dirname, abspath

def parse_args():
    """ parsing input command line parameters """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Worker Parser')
    parser.add_argument('--cmd_list_file', default='./cmd_lists.yaml', type=str, help="Commands list file location.")
    parser.add_argument('--session_name', '-sn', dest='sn', action='store', default='1', type=str, help="Session name for run.")
    parser.add_argument('--unavail_windows', '-uw', dest='uw', default=[0,1], type=list, help="Which window you don't want to run commands.")
    parser.add_argument('--user_name', '-un', dest='un', default=getpass.getuser(), type=str, help="User's name for window available check.")
    parser.add_argument('--memory_tolerance', '-mt', dest='mt', default=20, type=int, help="How many GBs of memory left can a new experiment run.")
    parser.add_argument('--wait_time', '-wt', dest='wt', default=5, type=int, help="Waiting seconds for each experiment.")
    
    return vars(parser.parse_args())

if __name__ == "__main__":
    pynvml.nvmlInit()
    GBUNIT = 1024 * 1024 * 1024
    args = parse_args()
    server = libtmux.Server()
    session = server.find_where({"session_name": args["sn"]})
    if not session:
        session = server.new_session(session_name=args["sn"])
    assert session
    with open(args['cmd_list_file'], "r") as f:
        cmd_list = f.readlines()
    cmd_list = [c.strip('\n') for c in cmd_list if len(c) > 1]
    print("All commands need to be printed:")
    new_cmd_list = []
    for cmd in cmd_list:
        if cmd[0] != '#':
            new_cmd_list.append(cmd)
        print(cmd)
    cmd_list =new_cmd_list
    idx = -1
    for i, cmd in enumerate(cmd_list):
        if cmd[0] == '$':
            if i!=0:
                print('GPU used, waiting for 5 minutes...')
                time.sleep(300)
            time_start = time.time()
            gpu_can_use = False
            while not gpu_can_use:
                max_gpu_used = 0
                for gpu_id in range(pynvml.nvmlDeviceGetCount()):
                    # if gpu_id<5:
                    #     continue
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    max_gpu_used = max(meminfo.used/GBUNIT,max_gpu_used)
                if max_gpu_used>1:
                    # if time.time()-time_start>3600*3 and not 'madtkd' in cmd_list[max(i-1,0)]:
                    #     gpu_can_use = True
                    #     os.system(os.path.join(dirname(abspath(__file__)),'kill_sc2.sh'))
                    #     print('clean sc2')
                    #     break
                    print('GPU used, waiting for 5 minutes...')
                    time.sleep(300)
                    
                else:
                    gpu_can_use = True
            idx=-1
            continue
        if i > 0:
            time.sleep(args["wt"])
        window_can_use=False
        while not window_can_use:
            idx += 1
            while(idx in args["uw"]):
                idx += 1
            if idx > 80:
                print("Window index too large, exit.")
                exit(0)
            print("Try window index", idx)
            try:
                w = session.new_window(attach=False, window_index=idx)
                time.sleep(10)
                print("Create new window index", idx)
            except:
                w = session.find_where({'window_index':str(idx)})
                # w = session.select_window(idx)
                print("Select existed windows index", idx)
            pm = w.list_panes()[0]
            lastLine = pm.cmd('capture-pane', '-p').stdout[-1]
            print('Last line is', lastLine)
            if args["un"]+'@' in lastLine and lastLine.endswith("$"):
                window_can_use=True
                print('Window available')
            else:
                print('Window occupied')
        cmd_can_run=False
        while not cmd_can_run:
            vm = psutil.virtual_memory()
            vm_avail = (vm.total-vm.used) / 1024 / 1024 / 1024
            print('Memory left', vm_avail, 'GB')
            if vm_avail >= args["mt"]:
                print('Momory enough')
                cmd_can_run=True
            else:
                print('Memory shortage, waiting for 2 minutes...')
                time.sleep(120)
        print("Run cmd:", cmd, "in window", idx)
        pm.send_keys(cmd, enter=True)
    print("All commands done.")