import os 
from script_run_train_rattnet import run_script
import yaml
from datetime import datetime
from utils import dump_info

# os.system('python.exe extract_latent_space_label.py --seq 00')
#models = ['sim_isr_backbone','sim_isr_1_attention','sim_isr_2_attention','sim_isr_3_attention','sim_isr_4_attention']

def attetion_session(**arg):

    session = arg['session']
    model   = arg['model']
    cfg     = arg['cfg']
    results  = arg['results']
    checkpoints = arg['checkpoints']
   
    # print("[SCRIPT FILE] "+ model)
    
    dest_network,dest_session = set_cfg(model,session,cfg)
    #sess = set_margin(session,dest_session,margin)
    
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
    
    # CMD = 'train_rattnet_knnv4.py'
    CMD = "inference_rattnet.py"
    TYPE_ = 'cross_val'
    root = "checkpoints/rattnet/"

    SEQUENCES = ['ex0','ex2','ex5','ex6','ex8']
    models = ['1bb_1a_norm','2bb_1a_norm','3bb_1a_norm','4bb_1a_norm','5bb_1a_norm']
    dest_model = 'network'
    
    attention_array = [0,1,2,3,4,5]
    
    plot = 0
    results = 'encoder_study.txt'    
    checkpoints = 'checkpoints'
    best_margin = 0.85
    
    for ex in SEQUENCES:
        
        dump_info(results,"",flag='a')

        for model in models:
            for att_value in attention_array:
                
                sess_name = '_'.join([ex,model,str(att_value)])
                checkpoints = os.path.join(checkpoints,sess_name)
          
                # Parameter to be changed in default files 
                cfg = {'network':{  'attention':{'n_layers':att_value,'train':True},
                                    'backbone':{'train':True},
                                    'outlayer':{'train':True}},
                        'session':{ 'loss_function':{'margin':best_margin},
                                    'val':{'fraction':round(1/3,2)}}
                }

                try:
                    # Build Session 
                    s = '%02d'%(int(ex[-1]))
                    session = TYPE_ + '_' + s

                    attetion_session(cmd = CMD,
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

