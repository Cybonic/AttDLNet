import os
import yaml

CMD = 'train_knn.py'

def run_script(**arg):

    model    = arg['model']
    session  = arg['session']
    cmd      = arg['cmd']

    train_arg_list  = [ '--model',model, 
                        '--sess_cfg',session
                        ]

    if 'results' in arg:
        value = arg['results']
        train_arg_list.append('--results')
        train_arg_list.append(value)

    if 'plot' in arg:
        value = str(arg['plot'])
        train_arg_list.append('--plot')
        train_arg_list.append(value)
    
    # Add pretrained if it exists 
    if "pretrained" in arg:
        value = arg['pretrained']
        if os.path.isfile(value + '.pth') ==  True:
            train_arg_list.append('--pretrained')
            train_arg_list.append(value)
    
    # Convert arguments to str line
    train_arg = ' '.join(train_arg_list)
    # Build Full terminal command 
    terminal_cmd_list = ['python.exe','-W','ignore' , cmd, train_arg]
    terminal_cmd      = ' '.join(terminal_cmd_list)
    
    print("\n\n======================================================")
    print("======================================================\n\n")

    print("[INF] $: %s\n"%(terminal_cmd))
    os.system(terminal_cmd)
  


def statup_session(**arg):

    #session = arg['session']
    models  = arg['model']
    root    = arg['root']
    sequences = arg['sequences']
    type_ = arg['type_'] 

    plot = arg['plot'] if 'plot' in arg else 0
 
    for model in models:
        for ex in sequences:
            # Build Argument 
            s = '%02d'%(int(ex[-1]))
            session = type_ + '_' + s
            pretrained =  root + model +'_' + session

            run_script( cmd = CMD,
                        model = model, 
                        session = session, 
                        pretrained = pretrained,
                        plot = plot
                    )


if __name__ == '__main__':
    TYPE_ = 'cross_val'
    root = "checkpoints/"

    session = 'cosine_small_session'
    model   = '2bb_1a_norm'

    network = yaml.load(open('model_cfg/'+ model + '.yaml'),Loader=yaml.FullLoader)

    with open('model_cfg/model.yaml', 'w') as file:
        documents = yaml.dump(network, file)

    run_script(cmd = CMD,model = 'model', session = session,plot=1)




