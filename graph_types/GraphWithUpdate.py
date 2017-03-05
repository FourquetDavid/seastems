'''
Created on 6 dec. 2012 

@author: David
inspired by Telmo Menezes's work : telmomenezes.com
''' 

""" 

abstract class that contains functions related to graphs   
four subclasses : (Un)directed(Un)weightedGraphWithUpdate

"""

class GraphWithUpdate():
       
    def add_edge(self,u,v,**args):
        raise NotImplementedError( "Should have implemented add_edge" ) 
    def isDirected(self):
        raise NotImplementedError( "Should have implemented isDirected" )    
    def isWeighted(self):
        raise NotImplementedError( "Should have implemented isWeighted" ) 
    def OrigDegree(self) : 
        raise NotImplementedError( "Should have implemented OrigDegree" )
    def NormalizedOrigDegree(self) : 
        raise NotImplementedError( "Should have implemented NormalizedOrigDegree" )   
    def OrigStrength(self) : 
        raise NotImplementedError( "Should have implemented OrigStrength" )
    def NormalizedOrigStrength(self) : 
        raise NotImplementedError( "Should have implemented NormalizedOrigStrength" ) 
    def OrigId(self) :
        raise NotImplementedError( "Should have implemented OrigId" )
    def NormalizedOrigId(self) :
        raise NotImplementedError( "Should have implemented NormalizedOrigId" )  
    def TargDegree(self) : 
        raise NotImplementedError( "Should have implemented TargDegree" )  
    def NormalizedTargDegree(self) : 
        raise NotImplementedError( "Should have implemented NormalizedTargDegree" )  
    def TargStrength(self) : 
        raise NotImplementedError( "Should have implemented TargStrength" )
    def NormalizedTargStrength(self) : 
        raise NotImplementedError( "Should have implemented NormalizedTargStrength" )  
    def TargId(self) : 
        raise NotImplementedError( "Should have implemented TargId" ) 
    def NormalizedTargId(self) : 
        raise NotImplementedError( "Should have implemented NormalizedTargId" ) 
    def Distance(self) :
        raise NotImplementedError( "Should have implemented Distance" )  
    def NormalizedDistance(self) :
        raise NotImplementedError( "Should have implemented NormalizedDistance" )
    def NumberOfNodes(self):
        raise NotImplementedError( "Should have implemented NumberOfNodes" )
    def NumberOfEdges(self):
        raise NotImplementedError( "Should have implemented NumberOfEdges" )   
    def MaxDegree(self):
        raise NotImplementedError( "Should have implemented MaxDegree" )
    def AverageDegree(self):
        raise NotImplementedError( "Should have implemented AverageDegree" )   
    def MaxStrength(self):
        raise NotImplementedError( "Should have implemented MaxStrength" )
    def AverageStrength(self):
        raise NotImplementedError( "Should have implemented AverageStrength" )
    def TotalWeight(self):
        raise NotImplementedError( "Should have implemented TotalWeight" ) 
    def AverageWeight(self):
        raise NotImplementedError( "Should have implemented AverageWeight" )
    def MaxWeight(self):
        raise NotImplementedError( "Should have implemented MaxWeight" )
    def MaxDistance(self) :
        raise NotImplementedError( "Should have implemented MaxDistance" )
    def AverageDistance(self) :
        raise NotImplementedError( "Should have implemented AverageDistance" )
    def TotalDistance(self) :
        raise NotImplementedError( "Should have implemented TotalDistance" )
    def Constant(self) :
        raise NotImplementedError( "Should have implemented Constant" )
    def Random(self) :
        raise NotImplementedError( "Should have implemented Random" )
    def OrigInDegree(self) : 
        raise NotImplementedError( "Should have implemented OrigInDegree" )
    def NormalizedOrigInDegree(self) : 
        raise NotImplementedError( "Should have implemented NormalizedOrigInDegree" )
    def OrigInStrength(self) : 
        raise NotImplementedError( "Should have implemented OrigInStrength" )
    def NormalizedOrigInStrength(self) : 
        raise NotImplementedError( "Should have implemented NormalizedOrigInStrength" )
    def OrigOutDegree(self) :
        raise NotImplementedError( "Should have implemented OrigOutDegree" )
    def NormalizedOrigOutDegree(self) :
        raise NotImplementedError( "Should have implemented NormalizedOrigOutDegree" )
    def OrigOutStrength(self) :
        raise NotImplementedError( "Should have implemented OrigOutStrength" )
    def NormalizedOrigOutStrength(self) :
        raise NotImplementedError( "Should have implemented NormalizedOrigOutStrength" )
    def TargInDegree(self) : 
        raise NotImplementedError( "Should have implemented TargInDegree" )
    def NormalizedTargInDegree(self) : 
        raise NotImplementedError( "Should have implemented NormalizedTargInDegree" )
    def TargInStrength(self) : 
        raise NotImplementedError( "Should have implemented TargInStrength" )
    def NormalizedTargInStrength(self) : 
        raise NotImplementedError( "Should have implemented NormalizedTargInStrength" )
    def TargOutDegree(self) :
        raise NotImplementedError( "Should have implemented TargOutDegree" )
    def NormalizedTargOutDegree(self) :
        raise NotImplementedError( "Should have implemented NormalizedTargOutDegree" )
    def TargOutStrength(self) :
        raise NotImplementedError( "Should have implemented TargOutStrength" )
    def NormalizedTargOutStrength(self) :
        raise NotImplementedError( "Should have implemented NormalizedTargOutStrength" )
    def DirectDistance(self) :
        raise NotImplementedError( "Should have implemented DirectDistance" )
    def NormalizedDirectDistance(self) : 
        raise NotImplementedError( "Should have implemented NormalizedDirectDistance" )
    def ReversedDistance(self) :
        raise NotImplementedError( "Should have implemented ReversedDistance" )   
    def NormalizedReversedDistance(self) :
        raise NotImplementedError( "Should have implemented NormalizedReversedDistance" )    
    def MaxInDegree(self):
        raise NotImplementedError( "Should have implemented MaxInDegree" )    
    def AverageInDegree(self):
        raise NotImplementedError( "Should have implemented AverageInDegree" )     
    def MaxOutDegree(self):
        raise NotImplementedError( "Should have implemented MaxOutDegree" )     
    def AverageOutDegree(self):
        raise NotImplementedError( "Should have implemented AverageOutDegree" )    
    def MaxInStrength(self):
        raise NotImplementedError( "Should have implemented MaxInStrength" )    
    def AverageInStrength(self):
        raise NotImplementedError( "Should have implemented AverageInStrength" )         
    def MaxOutStrength(self):
        raise NotImplementedError( "Should have implemented MaxOutStrength" )     
    def AverageOutStrength(self):
        raise NotImplementedError( "Should have implemented AverageOutStrength" ) 
    
    
    
        