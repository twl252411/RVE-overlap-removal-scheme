from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import string
import numpy as np
from math import *
import itertools
#
executeOnCaeStartup()
Mdb()
#
#---------------------------------------------------------------------------
#
Cublen, LayerT, IncRad, IncNum = 100.0, 0.0, 10.0, 120
CaeName = 'Sphere_Composites.cae'
#
#---------------------------------------------------------------------------
#
filename = 'points3d.txt'
pt1 = np.loadtxt(filename, delimiter=' ')
#
#---------------------------------------------------------------------------
#
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
s.ArcByCenterEnds(center=(0.0, 0.0), point1=(0.0, IncRad), point2=(0.0, -IncRad), direction=CLOCKWISE)
s.Line(point1=(0.0, -IncRad), point2=(0.0, IncRad))
p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['Part-1']
p.BaseSolidRevolve(sketch=s, angle=360.0, flipRevolveDirection=OFF)
s.unsetPrimaryObject()
del mdb.models['Model-1'].sketches['__profile__']
#
#---------------------------------------------------------------------------	
#
for i in range(len(pt1)):
    a = mdb.models['Model-1'].rootAssembly
    p = mdb.models['Model-1'].parts['Part-1']
    a.Instance(name='Part-1-'+str(i+1), part=p, dependent=OFF)
    a.translate(instanceList=('Part-1-'+str(i+1), ), vector=(pt1[i][0], pt1[i][1], pt1[i][2]))
#   
for i in range(1,len(pt1)):
    a = mdb.models['Model-1'].rootAssembly
    a.InstanceFromBooleanMerge(name='Part-New', instances=(a.instances['Part-1-1'], a.instances['Part-1-'+str(i+1)], ),
        keepIntersections=ON, originalInstances=DELETE, domain=GEOMETRY)
    a.deleteFeatures(('Part-New-1',))
    if i > 1:
    	del mdb.models['Model-1'].parts['Part-Temp']
    #
    mdb.models['Model-1'].parts.changeKey(fromName='Part-New', toName='Part-Temp')
    p = mdb.models['Model-1'].parts['Part-Temp']
    a.Instance(name='Part-1-1', part=p, dependent=OFF)
#
a = mdb.models['Model-1'].rootAssembly
a.deleteFeatures(('Part-1-1',))
mdb.models['Model-1'].parts.changeKey(fromName='Part-Temp', toName='Part-2')
del mdb.models['Model-1'].parts['Part-1']
#
a = mdb.models['Model-1'].rootAssembly
p = mdb.models['Model-1'].parts['Part-2']
a.Instance(name='Part-2-1', part=p, dependent=OFF)
a.Instance(name='Part-2-2', part=p, dependent=OFF)
#
#---------------------------------------------------------------------------	
#
s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=STANDALONE)
s1.rectangle(point1=(0.0, 0.0), point2=(Cublen,Cublen))
p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['Part-1']
p.BaseSolidExtrude(sketch=s1, depth=Cublen)
s1.unsetPrimaryObject()
del mdb.models['Model-1'].sketches['__profile__']
#
a = mdb.models['Model-1'].rootAssembly
p = mdb.models['Model-1'].parts['Part-1']
a.Instance(name='Part-1-1', part=p, dependent=OFF)
#
#---------------------------------------------------------------------------	
#
a = mdb.models['Model-1'].rootAssembly
a.InstanceFromBooleanCut(name='Part-3',instanceToBeCut=mdb.models['Model-1'].rootAssembly.instances['Part-2-1'], 
    cuttingInstances=(a.instances['Part-1-1'], ), originalInstances=DELETE)
a.InstanceFromBooleanCut(name='Part-4', instanceToBeCut=mdb.models['Model-1'].rootAssembly.instances['Part-2-2'],
    cuttingInstances=(a.instances['Part-3-1'], ), originalInstances=DELETE)
#
del mdb.models['Model-1'].parts['Part-2']
del mdb.models['Model-1'].parts['Part-3']
a = mdb.models['Model-1'].rootAssembly
a.deleteFeatures(('Part-4-1',))
mdb.models['Model-1'].parts.changeKey(fromName='Part-4', toName='Part-2')
#
a = mdb.models['Model-1'].rootAssembly
p = mdb.models['Model-1'].parts['Part-2']
a.Instance(name='Part-2-1', part=p, dependent=OFF)
a.translate(instanceList=('Part-2-1', ), vector=(LayerT/2.0, LayerT/2.0, LayerT/2.0))
#
#---------------------------------------------------------------------------
#
del mdb.models['Model-1'].parts['Part-1']
#
s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=STANDALONE)
s1.rectangle(point1=(0.0, 0.0), point2=(Cublen+LayerT,Cublen+LayerT))
p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['Part-1']
p.BaseSolidExtrude(sketch=s1, depth=Cublen+LayerT)
s1.unsetPrimaryObject()
del mdb.models['Model-1'].sketches['__profile__']
#
a = mdb.models['Model-1'].rootAssembly
p = mdb.models['Model-1'].parts['Part-1']
a.Instance(name='Part-1-1', part=p, dependent=OFF)
#
#---------------------------------------------------------------------------
#
a = mdb.models['Model-1'].rootAssembly
a.InstanceFromBooleanCut(name='Part-3', instanceToBeCut=mdb.models['Model-1'].rootAssembly.instances['Part-1-1'],
    cuttingInstances=(a.instances['Part-2-1'], ), originalInstances=DELETE)
#
del mdb.models['Model-1'].parts['Part-1']
a = mdb.models['Model-1'].rootAssembly
a.deleteFeatures(('Part-3-1',))
mdb.models['Model-1'].parts.changeKey(fromName='Part-3', toName='Part-1')
#
#---------------------------------------------------------------------------
#
a = mdb.models['Model-1'].rootAssembly
p = mdb.models['Model-1'].parts['Part-1']
a.Instance(name='Part-1-1', part=p, dependent=OFF)
a.translate(instanceList=('Part-1-1', ), vector=(-(Cublen+LayerT)/2.0, -(Cublen+LayerT)/2.0, -(Cublen+LayerT)/2.0))
p = mdb.models['Model-1'].parts['Part-2']
a.Instance(name='Part-2-1', part=p, dependent=OFF)
a.translate(instanceList=('Part-2-1', ), vector=(-Cublen/2.0, -Cublen/2.0, -Cublen/2.0))
#
#---------------------------------------------------------------------------
#
a = mdb.models['Model-1'].rootAssembly
a.InstanceFromBooleanMerge(name='Part-New', instances=(a.instances['Part-1-1'], a.instances['Part-2-1'], ),
    keepIntersections=ON, originalInstances=DELETE, domain=GEOMETRY)
a.deleteFeatures(('Part-New-1',))
del mdb.models['Model-1'].parts['Part-1']
del mdb.models['Model-1'].parts['Part-2']
mdb.models['Model-1'].parts.changeKey(fromName='Part-New', toName='Part-1')
#
a = mdb.models['Model-1'].rootAssembly
p = mdb.models['Model-1'].parts['Part-1']
a.Instance(name='Part-1-1', part=p, dependent=OFF)
#
#---------------------------------------------------------------------------
#
p.DatumCsysByThreePoints(name='Datum csys-1', coordSysType=CARTESIAN, origin=(0,0,0), point1=(1,0,0), point2=(0,1,0))
#
#---------------------------------------------------------------------------
#
p = mdb.models['Model-1'].parts['Part-1']
selcell = p.cells.findAt(((-(Cublen+LayerT)/2.0, -(Cublen+LayerT)/2.0, -(Cublen+LayerT)/2.0), ))
p.Set(cells=selcell, name='Set-M-1')
dic = p.getMassProperties(regions=p.sets['Set-M-1'].cells)
volum1 = dic['volume']
#
incCells = p.cells[0:0]
for icell in p.cells:
    if icell.index != selcell[0].index:
    	incCells += p.cells[icell.index:icell.index+1]
p.Set(cells=incCells, name='Set-M-2')
dic = p.getMassProperties(regions=p.sets['Set-M-2'].cells)
volum2 = dic['volume']
#
ParFrac = volum2/(volum2+volum1)
print ('The volume fraction of particles is')
print (ParFrac)
#
#---------------------------------------------------------------------------	
#
#CentersV = [0,0,0]
#p = mdb.models['Model-1'].parts['Part-1']
#SelectedAllCells = p.cells[0:0]
#for inum in range(IncNum):
#    #
#    Index = []
#    Index = Index + [inum]
#    if pt1[inum][3] != 0:
#        for jnum in range(IncNum,len(pt1)):
#            if pt1[inum][3] == pt1[jnum][3]:
#                Index = Index + [jnum]
#    #
#    SelectedCells = p.cells[0:0]
#    for iIndex in Index:
#        for icell in p.cells:
#            CentersV[0:3] = np.subtract(icell.pointOn[0][0:3], pt1[iIndex][0:3])
#            Distance = sqrt(CentersV[0]**2 + CentersV[1]**2 + CentersV[2]**2)
#            if Distance < IncRad or abs(Distance - IncRad) < 1E-6:
#                SelectedCells += p.cells[icell.index:icell.index+1]
#    #
#    SelectedAllCells += SelectedCells
#    SetName = 'Set-M-' + str(inum + 2)
#    p.Set(cells=SelectedCells, name=SetName)
##
#for icell in p.cells:
#    if icell not in SelectedAllCells:
#        p.Set(cells=p.cells[icell.index:icell.index+1], name='Set-M-1')
#
#---------------------------------------------------------------------------	
#	 
mdb.saveAs(pathName=CaeName)