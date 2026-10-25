# Wave hello

Your arm rises, waves hello three times, then folds back to rest. It takes about 10 seconds and
makes a gentle first app for a new arm.

## What it runs on

- The SO-101 follower arm, or the simulated SO-101 follower.
- A camera is optional. With one, the app counts the camera frames it sees while it waves.

Every move is slow and smooth: the arm never moves a joint faster than 45 degrees a second, and it
never jumps. At the end it glides back near where it started, and the arm's own slow fold to rest
does the last part.

## Run it from Robot App Store

Find Wave hello in Discover, install it, choose your arm, and click Run. Make sure the arm has room
to rise in front of it.

## Run it from a terminal

```
monorobot pull grand-wizard-create/wave-hello
monorobot robot bind hf_so101_follower --kwarg port=<port>
monorobot start wave-hello
```

Replace `<port>` with your arm's USB port.

## Stop it

In Robot App Store, press Space or Esc. In a terminal, press Ctrl+C: the arm folds back to rest.
Press Ctrl+C a second time to stop the arm where it is.

## License

Apache License 2.0. See `LICENSE`.
