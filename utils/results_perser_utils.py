
import argparse
import os
import pandas as pd

def model_parser(model):
    split = model.split('BB')
    backbone = int(split[0])
    attention = int(split[1].split('A')[0]) 
    return({'modelB':backbone,'modelA':attention})

def conv2dic(str_data):
    output = {}
    for elm in str_data: 
        key,str_value = elm.split(':')
        if key == 'session':
            value = str_value
            output[key] = value
        elif key == 'model':
            value = model_parser(str_value)
            for item_key, item_value in value.items():
                output[item_key] = item_value
        elif key == 'param':
            output[key] = int(str_value)
        elif key == 'epoch':
            output[key] = [int(v) for v in str_value.split('/')]
        else:
            output[key] = float(str_value)
    return(output)

def file_parser(file):

    if not os.path.isfile(file):
        print("[INF] Result file does not exist!")
        raise Exception

    df = pd.DataFrame() 
    for line in open(file):
        line = line.strip()
        if line =='': # Empty line
            continue
        line = line.split("||")
        if line[1] == '': # Line transition
            continue
        header = line[0]
        data = line[1].split(' ')
        # Convert from str to dictionairy
        persed_data = conv2dic(data)
        # convert to pandas
        persed_data_DF = pd.DataFrame(persed_data,columns = persed_data.keys())
        # append to global data frame
        df = df.append(persed_data_DF,ignore_index=True ) 
    return(df)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
      '--file', '-f',
      type=str,
      default = "results/margin_tunning.txt",
      required=False,
      help='Dataset to train with. No Default',
    )

    FLAGS, unparsed = parser.parse_known_args()
    # Get File
    file_to_parse = FLAGS.file
    # Parse file
    results = file_parser(file_to_parse)
    # Demo: get all data belonging to cross_val_00
    cross_val_00 = results[results.session == 'cross_val_00']
    print(cross_val_00)
