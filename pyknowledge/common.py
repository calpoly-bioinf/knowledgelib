import altair as alt

# use the url approach to keep notebook smaller
#url = 'Luminal_B_101_to_graph_fig1'
def histogram_1(source,url,x_var,color_var,row_var,opacity=0.3):
    source.to_json(url,orient='records')

    g = alt.Chart(url).transform_calculate(
        pct='100 / datum.Count'
    ).mark_area(
        opacity=opacity,
        interpolate='step'
    ).encode(
        alt.X('%s:Q'%x_var, bin=alt.Bin(maxbins=100)),
        alt.Y('sum(pct):Q', axis=alt.Axis(format='%'),stack=None),
        alt.Color('%s:N'%color_var),
        row='%s:N'%row_var
    )
    return g