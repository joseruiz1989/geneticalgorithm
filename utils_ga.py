import numpy as np
import pandas as pd
import random

def complete_bin(bin:list, numbers:int):
    new_bin = [0]*(numbers-len(bin))
    new_bin.extend(bin)
    return new_bin

def add_to_df(df, values):
    df_row = pd.DataFrame([values])
    df = df.append(df_row)
    return df

def rand_list(list):
    i = random.randrange(0, len(list))
    return list[i]

def int2bin(num:int):
	return [int(i) for i in list('{0:0b}'.format(num))]

def bin2int(bin:list):
	return int("".join(str(i) for i in bin),2)

