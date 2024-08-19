import numpy as np


class MyDecoder():

    def decode(self, qpos, qvel):
        # Note for now I am hardcoding the map from UR5e to Panda. Once we have the instantiation code and training code
        #  this can be sorted
        target_qpos_dim = 29
        target_qvel_dim = 27

        # Pad state.qpos and state.qvel to the target dimensions with zeros
        padded_qpos = np.pad(qpos, (0, target_qpos_dim - len(qpos)), mode='constant')
        padded_qvel = np.pad(qvel, (0, target_qvel_dim - len(qvel)), mode='constant')

        return padded_qpos, padded_qvel
