from datetime import datetime

def dump_info(file, text, flag='w'):
    now = datetime.now()
    current_time = now.strftime("%d|%H:%M:%S")
    
    f = open('results/' + file,flag)
    
    line = "{}||".format(now)

    if isinstance(text,dict):
        for key, values in text.items():
            line += "{}:{} ".format(key,values)
            
    elif isinstance(text,str):
        line += text
        #f.write(line)
    f.write(line + '\n')
    f.close()
    return(line)