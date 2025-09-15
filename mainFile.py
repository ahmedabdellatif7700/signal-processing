import numpy as np
# ======================= TxDSP 
class TxDSP:
    def __init__(self):
        pass

    def qammod(self):
        pass
    
# ======================= Channel 
class Channel:
    def __init__(self):
        pass

# ======================= RxDSP 
class RxDSP:
    def __init__(self):
        pass



# ======================= Simulation Orchestrator 
class SysOrch:
    """Run full QPSK simulation with LMS FFE or without equalization."""
    def __init__(self):
        pass

    def run(self):
        print("Running SysOrch")
        d = np.random.randint(0, 16, size=1000)


if __name__ == "__main__":
    sim = SysOrch()
    sim.run()
