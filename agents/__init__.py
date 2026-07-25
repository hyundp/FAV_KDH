from agents.aligen import AligenAgent
from agents.drift import DriftAgent
from agents.dsrl import DSRLAgent
from agents.fql import FQLAgent
from agents.fql_bc import FQLAgent as FQLBCAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.qam import QAMAgent
from agents.rebrac import ReBRACAgent
from agents.rlpd import RLPDAgent
from agents.sac import SACAgent

agents = dict(
    aligen=AligenAgent,
    drift=DriftAgent,
    dsrl=DSRLAgent,
    fql=FQLAgent,
    fql_bc=FQLBCAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    qam=QAMAgent,
    rebrac=ReBRACAgent,
    rlpd=RLPDAgent,
    sac=SACAgent,
)
