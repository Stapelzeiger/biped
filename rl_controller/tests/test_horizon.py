import jax as jp
import jax.numpy as jnp

HORIZON_STATE = 10 
state_history = None
state_size = 4

for i in range(5):
    # Initialize state history if needed.
    if state_history is None:
        state_history = jnp.zeros((HORIZON_STATE, state_size))


    current_state = jnp.array([1*(i+1), 2*(i+1), 3*(i+1), 4*(i+1)])
    # Update state history.
    state_history = jnp.roll(state_history, -1, axis=0)
    state_history = state_history.at[-1].set(current_state)

    print(i, state_history)

    print(state_history.ravel())

