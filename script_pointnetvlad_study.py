import os 
from script_run_train import run_script
import yaml
from datetime import datetime
from utils.utils import dump_info


def run_session(**arg):

    session = arg['session']
    cfg     = arg['cfg']
    results  = arg['results']
    checkpoints = arg['checkpoints']
    model = arg['model']

    
    dest_network,dest_session = set_cfg(model,session,cfg)
    
    run_script( cmd = arg['cmd'], 
                session = dest_session,
                model = dest_network,
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
    
    CMD = 'train_knn_pointnetvlad.py'
    TYPE_ = 'cross_val'
    root = "checkpoints/pointnetvlad"

    SEQUENCES = ['ex0','ex2','ex5','ex6','ex8']
    dest_model = 'network'
    
    batch_size= 1
    
    plot = 0
    results = 'pointnetvlad.txt'    
    checkpoints = 'checkpoints'
    best_margin = 0.85

    for ex in SEQUENCES:
        
        #text = "{}".format(ex)
        dump_info(results,"",flag='a')

                
        sess_name = '_'.join(['pointnetvlad',ex])
        checkpoints = os.path.join(checkpoints,sess_name)
        
        # Parameter to be changed in default files 
        cfg = {'session':{ 'loss_function':{'margin':best_margin},
                                    'test':{'fraction':round(1/5,2)},
                                    'train':{'report_val':2,
                                            'max_epochs':20,
                                            'fraction':round(1/5,2),
                                            'batch_size':1,
                                            'rotation': {'fraction':0}
                                        }
                        }
                }
        try:
            # Build Session 
            s = '%02d'%(int(ex[-1]))
            session = TYPE_ + '_' + s

            run_session(cmd = CMD,
                    model = 'pointnetvlad',
                    session = session,
                    results= results,
                    cfg = cfg,
                    plot = plot,
                    checkpoints = checkpoints)

        except KeyboardInterrupt:
            print("[WRN] Exiting APP")
            exit(0)
    
    print("***********************************************")

