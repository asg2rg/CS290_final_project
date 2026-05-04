## Code execution
The relevant code is located within /main. To train, run the training script main.py.

Arguments:

`--render` : if set, render the environment. Used primarily in eval  
`--save_path` : string; set the path here for overriding model save directory, otherwise training checkpoint will be in the working directory where the script is launched (/main)  
`--unclamp` : if set, the action outputs are not clamped to a legal range, but the env penalizes the agent for the agent to learn the range. Tends to produces more stable behavior, but may take longer to converge. Recommendation is to run both --unclamp and without --unclamp for each experiment. If this is not set, the checkpoints and logs will have _clamped added to the prefix  
`--discrete` : unused.  
`--eval` : if set, this run will be in eval mode  
`--exp-name` : string; set the experiment name prefix.

During the run, if the car is able to reach the max steps per episode without being terminated, and it has been 30 episodes since the last time a training checkpoint was saved, a new checkpoint is saved so that performance can be evaluated even during training.

### Example usage
To train and save the logs and checkpoint to files with the prefix `multi_clamped`, run `python main.py --exp-name multi`. To eval using this checkpoint, run `python main.py --exp-name multi --eval` (and add `--render` to visualize).
