import configparser

def getConfigBySectionKey(section,key):
    rlt = ''
    config = configparser.ConfigParser()
    config.read('config.ini')
    rlt = config[section][key]
    return rlt