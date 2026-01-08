okay so focus

our goal is to get the camera to follow the end effect for now

we can use p.getLinkState() for that, with a link id and robot id

the problem here is to structure the code in such a way that you can actually get the link state from the Camera, or at least feed it into the camera from elsewhere

soooooooo

let's thunk

we can go the smart way about this and place the logic for getting the camera link state from the camera itself. That didn't work last time apparently because the ink state and the robot id needs to be established by the environment first, meanwhile the envvironment nees the camera to exist first.

With the current structure, that logically means that it's not possible to save the link and robot id to the camera right on instantiation.

However, what if i just do a little cheaty maneuver and modify the camera once it's passed into the environment?

eehhhhh it's possible but that would technically be an architectural no-no and I'm not happy about that.

the right way to go about it would be to move the logic to a completely separate spot, but I frankly dislike that too because for this project scale it's oo much work.

so for now, let's assume that I went for the "feeding link and robot id as param" approach

what does that mean for the rest of this development?

that depends on what I need to do next

I need to establish logic for actually setting up the parameter from the terminal. I need to pick between all the modes.

Actually, in a way, it might be more logical to start off with that, especially because there's multiple modes to worry about, not just the default and the hand on eye mode.

Best way to go about it is probably to first setup a dict.

okay the arg parser would probably only work with strings so that simplifies things a little bit (by making them really fucking rudimentary and ugly but I digress)

so in that case I need to worry about two things 

setting camera position and instantiating the right camera type (!!)

other than that this should be a one day job ffr.

______________________________


Okay so 

we have the issue that the grasp detector is not correctly matching up its detected grasps to the real world. if I look at the camera output files, things do make sense, so it's just a matter of translating the right thing.

the camera has roughyly the same orientation, but a completely differnet position.


I think te problem lies in GraspGenerator:grasp_to_robot_frame