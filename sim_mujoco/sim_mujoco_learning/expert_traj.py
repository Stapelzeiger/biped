from scipy.interpolate import interp1d

class ExpertTrajectory:
    # TODO: Match columns with each of these
    
    TIME_COL = "time"
    QPOS_COL = ["..."]
    QVEL_COL = [""]
    OBS_COL = [""]
    ACTION_COL = [""]

    def __init__(self, exp_data):
        """A class to handle referencing expert data

        Args:
            exp_data (pd.DataFrame): a dataframe containing the expert data
        """
        self.time = exp_data[ExpertTrajectory.TIME_COL].to_numpy()
        self.time -= self.time[0]
        self.qpos = exp_data[ExpertTrajectory.QPOS_COL].to_numpy()
        self.qvel = exp_data[ExpertTrajectory.QVEL_COL].to_numpy()
        self.obs = exp_data[ExpertTrajectory.OBS_COL].to_numpy()
        self.action = exp_data[ExpertTrajectory.ACTION_COL].to_numpy()

        self.interp_qpos = interp1d(self.time, self.qpos)
        self.interp_qvel = interp1d(self.time, self.qvel)
        self.interp_obs = interp1d(self.time, self.obs)
        self.interp_action = interp1d(self.time, self.action)
    
    def get_qpos_time(self, time):
        """Get the joint positions of the expert at a given time

        Args:
            time (float): time at which to query expert trajectory

        Returns:
            array-like: joint positions of the expert at a given time
        """
        return self.interp_qpos(time)

    def get_qpos_step(self, step):
        """Get the joint positions of the expert at a given step

        Args:
            step (int): step at which to query expert trajectory

        Returns:
            array-like: joint positions of the expert at a given step
        """
        return self.qpos[step, :]
    
    def get_qvel_time(self, time):
        """Get the joint velocities of the expert at a given time

        Args:
            time (float): time at which to query expert trajectory

        Returns:
            array-like: joint velocities of the expert at a given time
        """
        return self.interp_qvel(time)

    def get_qvel_step(self, step):
        """Get the joint velocity of the expert at a given step

        Args:
            step (int): step at which to query expert trajectory

        Returns:
            array-like: joint velocities of the expert at a given step
        """
        return self.qvel[step, :]
    
    def get_obs_time(self, time):
        """Get the observation of the expert at a given time

        Args:
            time (float): time at which to query expert trajectory

        Returns:
            array-like: observations of the expert at a given time
        """
        return self.interp_obs(time)

    def get_obs_step(self, step):
        """Get the observation of the expert at a given step

        Args:
            step (int): step at which to query expert trajectory

        Returns:
            array-like: observations of the expert at a given step
        """
        return self.obs[step, :]
    
    def get_action_time(self, time):
        """Get the action of the expert at a given time

        Args:
            time (float): time at which to query expert trajectory

        Returns:
            array-like: action of the expert at a given time
        """
        return self.interp_action(time)

    def get_action_step(self, step):
        """Get the action of the expert at a given step

        Args:
            step (int): step at which to query expert trajectory

        Returns:
            array-like: action of the expert at a given step
        """
        return self.action[step, :]
    