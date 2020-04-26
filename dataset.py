#####CREATING DATASET ######


###Manufacturing plans
three_d_plans = pd.read_csv(r"~/Desktop/manufacturing_plans/3D_printing.csv")
machining_plans = pd.read_csv(r"~/Desktop/manufacturing_plans/machining_plan.csv")
welding_plans = pd.read_csv(r"~/Desktop/manufacturing_plans/welding_plan.csv")
casting_plans = pd.read_csv(r"~/Desktop/manufacturing_plans/casting_plan.csv")

####miller code csv
m_3 = pd.read_csv(r"~/Desktop/manufacturing_plans/m_code/3D_printing_results.csv")
m_casting = pd.read_csv(r"~/Desktop/manufacturing_plans/m_code/casting_results.csv")
m_machining = pd.read_csv(r"~/Desktop/manufacturing_plans/m_code/machining_results.csv")
m_welding = pd.read_csv(r"~/Desktop/manufacturing_plans/m_code/Welding_results.csv")


#data cleaning 

m_3 = m_3.drop([
    'num_faces', 
    'num_wires',
    'num_edges',
    'Geom_Plane',
    'Geom_CylindricalSurface',
    'Geom_ConicalSurface',
    'Geom_SphericalSurface',
    'Geom_ToroidalSurface',
    'Geom_BezierSurface',
    'Geom_BSplineSurface',
    'Geom_SurfaceOfRevolution',
    'Geom_SurfaceOfLinearExtrusion',
    'Geom_Surface',
    'Geom_OffsetSurface',
   'Geom_Surface_Unknown',
    'Geom_Wire_Unknown',
    'Geom_Line',
    'Geom_Circle',
    'Geom_Ellipse',
    'Geom_Hyperbola',
    'Geom_Parabola',
    'Geom_BezierCurve',
    'Geom_BSplineCurve',
    'Geom_Curve',
    'Geom_Edge_Unknown'], axis = 1)

m_3 = m_3.loc[:, ~m_3.columns.str.contains('^Unnamed')]
m_3["filename"] = m_3["filename"].str.replace("./step-files/", "")
m_3['Process']='3D-Printing'

##

m_casting.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

m_casting = m_casting.drop([
    'num_faces', 
    'num_wires',
    'num_edges',
    'Geom_Plane',
    'Geom_CylindricalSurface',
    'Geom_ConicalSurface',
    'Geom_SphericalSurface',
    'Geom_ToroidalSurface',
    'Geom_BezierSurface',
    'Geom_BSplineSurface',
    'Geom_SurfaceOfRevolution',
    'Geom_SurfaceOfLinearExtrusion',
    'Geom_Surface',
    'Geom_OffsetSurface',
   'Geom_Surface_Unknown',
    'Geom_Wire_Unknown',
    'Geom_Line',
    'Geom_Circle',
    'Geom_Ellipse',
    'Geom_Hyperbola',
    'Geom_Parabola',
    'Geom_BezierCurve',
    'Geom_BSplineCurve',
    'Geom_Curve',
    'Geom_Edge_Unknown'], axis = 1)

m_casting = m_casting.loc[:, ~m_casting.columns.str.contains('^Unnamed')]
m_casting["filename"] = m_casting["filename"].str.replace("./step-files/", "")
m_casting['Process']='Casting'
##

m_machining = m_machining.drop([
    'num_faces', 
    'num_wires',
    'num_edges',
    'Geom_Plane',
    'Geom_CylindricalSurface',
    'Geom_ConicalSurface',
    'Geom_SphericalSurface',
    'Geom_ToroidalSurface',
    'Geom_BezierSurface',
    'Geom_BSplineSurface',
    'Geom_SurfaceOfRevolution',
    'Geom_SurfaceOfLinearExtrusion',
    'Geom_Surface',
    'Geom_OffsetSurface',
   'Geom_Surface_Unknown',
    'Geom_Wire_Unknown',
    'Geom_Line',
    'Geom_Circle',
    'Geom_Ellipse',
    'Geom_Hyperbola',
    'Geom_Parabola',
    'Geom_BezierCurve',
    'Geom_BSplineCurve',
    'Geom_Curve',
    'Geom_Edge_Unknown'], axis = 1)

m_machining = m_machining.loc[:, ~m_machining.columns.str.contains('^Unnamed')]
m_machining["filename"] = m_machining["filename"].str.replace("./step-files/", "")
m_machining['Process']='Machining'

##
m_welding = m_welding.drop([
    'num_faces', 
    'num_wires',
    'num_edges',
    'Geom_Plane',
    'Geom_CylindricalSurface',
    'Geom_ConicalSurface',
    'Geom_SphericalSurface',
    'Geom_ToroidalSurface',
    'Geom_BezierSurface',
    'Geom_BSplineSurface',
    'Geom_SurfaceOfRevolution',
    'Geom_SurfaceOfLinearExtrusion',
    'Geom_Surface',
    'Geom_OffsetSurface',
   'Geom_Surface_Unknown',
    'Geom_Wire_Unknown',
    'Geom_Line',
    'Geom_Circle',
    'Geom_Ellipse',
    'Geom_Hyperbola',
    'Geom_Parabola',
    'Geom_BezierCurve',
    'Geom_BSplineCurve',
    'Geom_Curve',
    'Geom_Edge_Unknown'], axis = 1)

m_welding = m_welding.loc[:, ~m_welding.columns.str.contains('^Unnamed')]
m_welding["filename"] = m_welding["filename"].str.replace("./step-files/", "")
m_welding['Process']='Welding'

#for col in data.columns: 
#    print(col) 




###3-D printing
#[4,5] formula 
def time_3D(row):
    for i in row:
        return row['volume_properties'] / 452.16

m_3['time_3D'] = m_3.apply(time_3D, axis=1)

def total_3D_time(row):
    for i in row:
        return row['time_3D'] + 0.1 + 3 + 2 + 0.5 + 3 + 5

m_3['total_3D_time'] = m_3.apply(total_3D_time, axis=1)

def cost_3D(row):
    for i in row: 
        return 93.4244 + 3.43239 * (row['total_3D_time']/60)

m_3['cost_3D'] = m_3.apply(cost_3D, axis = 1)

###Welding

def time_welding(row):
    for i in row:
        return row['length'] * 0.03

m_welding['time_welding'] = m_welding.apply(time_welding, axis=1)

def total_welding_time(row):
    for i in row:
        return row['time_welding'] + 0.5 + 5 + 0.5 + 0.2 + 2 + 5

m_welding['total_welding_time'] = m_welding.apply(total_welding_time, axis=1)

def cost_welding(row):
    for i in row:
        return 0.05942 + 30.6621 * (row['total_welding_time']/60)

m_welding['welding_cost'] = m_welding.apply(cost_welding, axis=1)


###Casting
### need new equations

def time_casting1(row):
    c=2
    for i in row:
        return ((.0027*row['volume_properties'])*.9*640) / 150

m_casting['time_casting1'] = m_casting.apply(time_casting1, axis=1)

def time_casting20(row):
    c=2
    for i in row:
        return c * (row['volume_properties']/row['surface_properties'])**2

m_casting['time_casting20'] = m_casting.apply(time_casting20, axis=1)

def time_casting23(row):
    c=2
    for i in row:
        return c * (row['volume_properties']/row['surface_properties'])**2

m_casting['time_casting23'] = m_casting.apply(time_casting23, axis=1)


def time_casting24(row):
    for i in row:
        return row['surface_properties'] * 0.0005

m_casting['time_casting24'] = m_casting.apply(time_casting24, axis=1)

def total_casting_time(row):
    for i in row:
        return row['time_casting1'] + row['time_casting20'] + row['time_casting23'] + row['time_casting24'] + 0.5 + 0.5 + 1 + 1 + 0.5 + 0.5 + 1 + 1 + 0.5 + 0.5 + 0.25 + 5 + 0.5 + 0.5 + 0.5 + 1 + 2 + 2 + 2 + 2

m_casting['total_casting_time'] = m_casting.apply(total_casting_time, axis=1)

def casting_cost(row):
    for i in row:
        return 1008.11 + 27.050 * (row['total_casting_time']/60)

m_casting['casting_cost'] = m_casting.apply(casting_cost, axis=1)


##machining 
#Feed_Rate = 2.601*10^7 

def time_machining(row):
    for i in row:
        return row['length']/ 7112

m_machining['time_machining'] = m_machining.apply(time_machining, axis=1)

def total_machining_time(row):
    for i in row:
        return row['time_machining'] + 2.5 + 1.5 + 1

m_machining['total_machining_time'] = m_machining.apply(total_machining_time, axis=1)

def machining_cost(row):
    for i in row:
        return 2.009 + 30.6621 * (row['total_machining_time']/60)

m_machining['machining_cost'] = m_machining.apply(machining_cost, axis=1)


x_feat = m_welding.append(m_machining, ignore_index=True, verify_integrity=False, sort=None)
x_feat = x_feat.append(m_casting, ignore_index=True, verify_integrity=False, sort=None)
x_feat = x_feat.append(m_3, ignore_index=True, verify_integrity=False, sort=None)

#need to fill in empty times w/ zeros 
x_feat = x_feat.fillna(0)
x_feat = x_feat[(x_feat.T != 0).any()]
#rearranging columns for easier view 
x_feat = x_feat[['filename', 'Process', 'length', 'width', 'height', 'linear_properties', 'surface_properties', 'volume_properties', 'Geom_Vertex_Unknown', 'time_welding', 'total_welding_time','welding_cost','time_machining', 'total_machining_time', 'machining_cost', 'time_casting1', 'time_casting20', 'time_casting23', 'time_casting24', 'total_casting_time', 'casting_cost', 'time_3D', 'total_3D_time', 'cost_3D']]
#printing out csv with formula time and total time 
x_feat = x_feat.to_csv(r"~/Desktop/capstone/dataset_2020.csv", index=True)


