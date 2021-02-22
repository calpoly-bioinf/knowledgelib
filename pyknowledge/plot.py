import altair as alt
import numpy as np

def prepare_url(source,url):
    source.to_json(url,orient='records')

# use the url approach to keep notebook smaller
def histogram_1(url,x_var,color_var,row_var,opacity=0.3,file=None):
    g = alt.Chart(url).transform_calculate(
        pct='1 / datum.Count'
    ).mark_area(
        opacity=opacity,
        interpolate='step'
    ).encode(
        alt.X('%s:Q'%x_var, bin=alt.Bin(maxbins=100)),
        alt.Y('sum(pct):Q', axis=alt.Axis(format='%'),stack=None),
        alt.Color('%s:N'%color_var),
        row='%s:N'%row_var
    )
    if file is not None:
        g.save(file)
    return g

# get things in order so that LumA - LumB and LumB - LumA are reordered consistently
def reorder_distance(distance0,first,second,sign=1):
    distance0_12 = distance0.loc[distance0['label1_label2'] == f'{first} - {second}'].copy()
    if first != second:
        add = distance0.loc[distance0['label1_label2'] == f'{second} - {first}'].copy()
        cols = distance0.select_dtypes(include=np.number).columns.tolist()
        add[cols] = sign*add[cols]
        distance0_12 = distance0_12.append(add)
    distance0_12['label1_label2_ordered'] = f'{first} - {second}'
    return distance0_12

def prepare_source1(df,distance1,constant,others):
    distance1_12 = reorder_distance(distance1,constant,constant)
    for n in others:
        if n != constant:
            distance1_12 = distance1_12.append(reorder_distance(distance1,constant,n))

    return distance1_12.reset_index()
