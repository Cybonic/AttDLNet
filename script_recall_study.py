import os 
from script_run_train import run_script
import yaml
from datetime import datetime
from utils.utils import dump_info


def recall_session(**arg):

    session = arg['session']
    model   = arg['model']
    cfg     = arg['cfg']
    results  = arg['results']
    checkpoints = arg['checkpoints']

    
    dest_network,dest_session = set_cfg(model,session,cfg)
    
    run_script( cmd = arg['cmd'],
                model = dest_network, 
                session = dest_session,
                plot = plot,
                results = results,
                checkpoints = checkpoints
            )

def set_cfg(network,src_session,cfg):

    dest_session = 'session'
    dest_network = 'network'

    ##### SESSION STUFF ####
    # Load session file
    session_dest_path = 'sessions/'+ dest_session + '.yaml'
    session_src_path  = 'sessions/'+ src_session + '.yaml'
    # Delete destination YMAL file 
    if os.path.isfile(session_dest_path):
        os.remove(session_dest_path)
    # Load src data
    session = yaml.load(open(session_src_path),Loader=yaml.FullLoader)
    # Copy to dst data
    if 'session' in cfg:
        # train = session['train']
        session_cfg = cfg['session'] # src session
        session['name'] = src_session # add src session to dest
        for key, value in session_cfg.items():
            for subkey, subvalue in value.items():
                session[key][subkey] = subvalue
    # Dump data to destination file
    with open(session_dest_path, 'w') as file:
        documents = yaml.dump(session, file)
    # -----------------------------------------------------------------
    #### NETWORK STUFF ####
    # Load Network file 
    network_dest_path = 'model_cfg/'+ dest_model+ '.yaml'
    network_scr_path = 'model_cfg/'+ network+ '.yaml'
    # Delete previous existing dest files
    if os.path.isfile(network_dest_path):
        os.remove(network_dest_path)
    # Load stuff from source YMAL file 
    network = yaml.load(open(network_scr_path),Loader=yaml.FullLoader)

    if 'network' in cfg:
        network_cfg = cfg['network']
        for key, value in network_cfg.items():
            for subkey,subvalue in value.items():
                network[key][subkey] = subvalue
    # Dump data to destination file
    with open(network_dest_path, 'w') as file:
        documents = yaml.dump(network, file)
    
    return(dest_network,dest_session)



if __name__ == '__main__':
    
    TYPE_ = 'cross_val'
    root = "checkpoints/"
    CMD = 'inference_pointnetvlad.py'

    SEQUENCES = ['ex0','ex2','ex5','ex6','ex8']
    models = ['pointnetvlad']
    
    dest_model = 'network'
    top_cand = [1,2,4,6,8,10,20,30,40,50,60,80,100,200,400]
    plot = 0
    results = 'pointnetvlad_recall.txt'
    best_margin = 0.85 
    checkpoints = 'checkpoints'   

    for ex in SEQUENCES:
        dump_info(results,"",flag='a')
        for model in models:
            for value in top_cand:

                try:
                    # Build Argument 
                    s = '%02d'%(int(ex[-1]))
                    session = TYPE_ + '_' + s

                    cfg = {'session':{ 'loss_function':{'margin':best_margin},
                                        'retrieval':{'top_cand':value},
                                        'test':{'fraction':round(1/5,2)}},
                                        'train':{'report_val':2,
                                            'max_epochs':20,
                                            'fraction':round(1/5,2),
                                            'batch_size':3
                                }
                    }

                    # Build Session 
                    s = '%02d'%(int(ex[-1]))
                    session = TYPE_ + '_' + s

                    recall_session(cmd = CMD,
                                    model = model,
                                    session = session,
                                    results= results,
                                    cfg = cfg,
                                    plot = plot,
                                    checkpoints = checkpoints)

                except KeyboardInterrupt:
                    print("[WRN] Exiting APP")
                    exit(0)
            
            print("***********************************************")

