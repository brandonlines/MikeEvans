"""Normalized expression measurements and optional per-session range calibration."""
import numpy as np
from puppet import canonical

LEFT=[70,63,105,66,107,55,65,52,53,46]
RIGHT=[336,296,334,293,300,285,295,282,283,276]

def metrics(points):
    p,_,_,_=canonical(points)
    heights=[p[[33,133],1].mean()-p[LEFT,1].mean(),
             p[[263,362],1].mean()-p[RIGHT,1].mean()]
    openings=[abs(p[159,1]-p[145,1]),abs(p[386,1]-p[374,1])]
    return np.array(heights+openings,dtype=np.float32)

class ExpressionMap:
    def __init__(self):self.up=np.array([.06,.06]);self.down=np.array([.04,.04]);self.closed=None
    def calibrate(self,neutral,raised,furrowed,closed):
        base=metrics(neutral)
        self.up=np.clip(metrics(raised)[:2]-base[:2],.025,.2)
        self.down=np.clip(base[:2]-metrics(furrowed)[:2],.025,.2)
        self.closed=metrics(closed)[2:]
    def values(self,current,neutral):
        value=metrics(current);base=metrics(neutral);delta=value[:2]-base[:2]
        brows=np.clip(delta/np.where(delta>=0,self.up,self.down),-1,1)
        floor=base[2:]*.2 if self.closed is None else self.closed
        closure=np.clip((base[2:]-value[2:])/np.maximum(base[2:]-floor,.015),0,1)
        return brows,closure
