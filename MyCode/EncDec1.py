import numpy as np
import torch


class EncDec1():

    def decode_env_side(self, qpos, qvel, qpos_target_shape, qvel_target_shape):
        print("DECODING")
        # Note for now I am hardcoding the map from UR5e to Panda. Once we have the instantiation code and training code
        #  this can be sorted
        # Ensure the target shapes are tuples
        if not isinstance(qpos_target_shape, tuple) or not isinstance(qvel_target_shape, tuple):
            raise ValueError("Target shapes must be tuples.")

        # Calculate the required padding for qpos and qvel
        qpos_padding = (0, qpos_target_shape[0] - qpos.shape[0])  # Padding for qpos
        qvel_padding = (0, qvel_target_shape[0] - qvel.shape[0])  # Padding for qvel

        # Check if any padding is necessary
        if qpos_padding[1] > 0:
            # Pad qpos to the target dimensions with zeros
            padded_qpos = np.pad(qpos, qpos_padding, mode='constant')
            print(
                f"Converted original qpos shape from {qpos.shape} to {padded_qpos.shape} by adding {qpos_padding[1]} zeros.")
        else:
            # No padding needed, use the original qpos
            padded_qpos = qpos

        if qvel_padding[1] > 0:
            # Pad qvel to the target dimensions with zeros
            padded_qvel = np.pad(qvel, qvel_padding, mode='constant')
            print(
                f"Converted original qvel shape from {qvel.shape} to {padded_qvel.shape} by adding {qvel_padding[1]} zeros.")
        else:
            # No padding needed, use the original qvel
            padded_qvel = qvel

        return padded_qpos, padded_qvel


    def encode_obs_model_side(self, data):
        """
            Pads the observation data so that the UR5e robot's data matches the Panda's format.
        """
        # Check if the data corresponds to the UR5e (based on the shape of the gripper_states or joint_states)
        if data['obs']['gripper_states'].shape[-1] == 6:
            # Truncate gripper_states to match Panda's format
            data['obs']['gripper_states'] = data['obs']['gripper_states'][..., :2]

            # Pad joint_states to match Panda's format
            padding = (0, 1)  # Pad 1 zero to the end along the last dimension
            data['obs']['joint_states'] = torch.nn.functional.pad(data['obs']['joint_states'], padding)

        # Return the padded data
        return data

