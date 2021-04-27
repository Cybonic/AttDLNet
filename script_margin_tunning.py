import os 
from script_run_train_rattnet import run_script
import yaml
from datetime import datetime
from utils import dump_info

def dump_conf_info(file, cfg, flag='w'):

    f = open('results/' + file,flag)

    if isinstance(cfg,dict):
        for key, values in cfg.items():
            line = "{} {}".format(key,values)
            f.write(line + '\n')
    elif isinstance(cfg,str):
        f.write(cfg + '\n')
    
    f.close()

def statup_session(**arg):

    session = arg['session']
    model   = arg['model']
    margin  = arg['margin']
    results  = arg['results']
    cmd = arg['cmd']

    print("[SCRIPT FILE] "+ model)
    
    dest_network = 'network'
    dest_session = 'session'
    
    network_cfg(model,dest_network)
    sess = set_margin(session,dest_session,margin)
    
    run_script( cmd = cmd,
                model = dest_network, 
                session = dest_session,
                plot = plot,
                results = results
            )

def set_margin(src,dest,value):

    dest_path = 'sessions/'+ dest+ '.yaml'
    src_path  = 'sessions/'+ src+ '.yaml'
    
    if os.path.isfile(dest_path):
        os.remove(dest_path)
    session = yaml.load(open(src_path),Loader=yaml.FullLoader)
    session['loss_function']['margin'] = value
    session['train']['max_epochs'] = 21
    session['train']['report_val'] = 3
    session['train']['fraction'] = round(1/5,2)
    session['val']['fraction'] = round(1/3,2)
    
    with open(dest_path, 'w') as file:
        documents = yaml.dump(session, file)
    
    return(session)

def network_cfg(src_model,dest_model):
    
    if os.path.isfile('model_cfg/'+ dest_model+ '.yaml'):
        os.remove('model_cfg/'+ dest_model+ '.yaml')

    network = yaml.load(open('model_cfg/'+ model + '.yaml'),Loader=yaml.FullLoader)
    #network_cnf.yaml
    network['backbone']['train']  = True
    network['attention']['train'] = True
    network['outlayer']['train']  = True

    with open('model_cfg/' + dest_model + '.yaml', 'w') as file:
        documents = yaml.dump(network, file)

    return(network)


if __name__ == '__main__':
    
    TYPE_ = 'cross_val'
    root = "checkpoints/rattnet/"
    CMD = 'train_rattnet_knnv3.py'

    SEQUENCES = ['ex0','ex2','ex5','ex6','ex8']
    models = ['1bb_1a_norm','2bb_1a_norm','3bb_1a_norm','4bb_1a_norm','5bb_1a_norm']
    
    dest_model = 'network'
    margin_array = [0.0,0.3,0.5,0.7,0.8,0.85,0.9,0.95]
    plot = 0
    results = 'margin_tunning.txt'    

    for ex in SEQUENCES:
        dump_info(results,"",flag='a')
        for model in models:
            for margin in margin_array:

                try:
                    # Build Argument 
                    s = '%02d'%(int(ex[-1]))
                    session = TYPE_ + '_' + s

                    statup_session(
                                cmd = CMD,
                                model = model,
                                session = session,
                                results= results,
                                margin = margin,
                                plot = plot)

                except KeyboardInterrupt:
                    print("[WRN] Exiting APP")
                    exit(0)
            
            print("***********************************************")

