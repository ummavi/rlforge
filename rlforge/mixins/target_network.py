from rlforge.mixins.base_mixin import BaseMixin

class TargetNetworkMX(BaseMixin):
    """Target Network Mixin 
    """
    def __init__(self, target_network_update_freq):
        BaseMixin.__init__(self)

        self.target_network_update_freq = target_network_update_freq

        #Create a target network on initialization.
        self.create_target_network()

        self.post_step_hooks.append(self.update_target_network)

    def create_target_network(self):
        """Clone the network
        """
        self.target_model =  self.model.clone()

    def update_target_network(self, global_step_ts, step_data):
        """update the target network's weights every few steps.
        post_step_hook Parameters:
            global_step_ts (int)
            step_data (tuple): (s,a,r,s',done)
        """
        if global_step_ts%self.target_network_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
