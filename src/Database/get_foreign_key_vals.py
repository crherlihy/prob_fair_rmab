'''get_foreign_key_vals.py: get next simulation key values from database
'''
import configparser

import src.utils as simutils
from collections import OrderedDict
from src.Database.read import DBReader

def main(params: OrderedDict):
    '''
    Get foreign key values
    
    :param params: DBReader parameters to connect to the database
    :type params: OrderedDict
    :return: key values
    :rtype: str

    '''
    
    reader = DBReader(params)
    
    statement = """SELECT MAX(s.auto_id) from simulations as s;"""
    reader.cursor.execute(statement)
    out = reader.cursor.fetchall()[0]
    key_vals = "".join([str(int(x)+1) if x is not None else str(1) for x in out])
    reader.cursor.close()
    reader.conn.close()
    if not key_vals:
        key_vals = "0"  
    return key_vals


if __name__ == "__main__":
    arg_groups = simutils.get_args(argv=None)

    config_filename = vars(arg_groups['general'])['config_file']
    #level = '..'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_filename)
    
    config['database']['db_ip'] = vars(arg_groups['database'])['db_ip']
    key_vals = main(OrderedDict({k.upper(): v for k,v in config['database'].items()}))
    print(key_vals)
    