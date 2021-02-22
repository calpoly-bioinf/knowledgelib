def divide_chunks(l, n): 
    r = []
    # looping till length l 
    for i in range(0, len(l), n):  
        r.append(l[i:i + n])
    return r

def get_pair_inxs(n,r=0):
    pair_ixs = []
    for i in range(n):
        for j in range(i+r,n):
            pair_ixs.append((i,j))
    return pair_ixs

def calc_len_chunk(n,nsplit):
    return round(n/nsplit)

def add_labels(df,distance0,label_col='Subtype'):
    label1 = []
    label2 = []
    for index1,index2 in distance0.index:
        label1.append(df.loc[index1,label_col])
        label2.append(df.loc[index2,label_col])
    distance0['label1'] = label1
    distance0['label2'] = label2
    distance0['label1_label2'] = distance0['label1'] + " - "+distance0['label2']
    return distance0

def subset_columns(df,L):
    """
    L is a list (or series) of genes
    """
    return df.loc[:,df.columns.isin(L)]