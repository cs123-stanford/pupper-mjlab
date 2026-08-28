=========
Changelog
=========

Upcoming version (not yet released)
-----------------------------------

Added
^^^^^

- Added ``Mjlab-MixedGaitsJump-Flat-Pupper-v3``: MixedGaits with an
  insertable one-shot jump ("mixed gait + jump"). A trigger -- a per-episode
  schedule in training, a gamepad button on the robot -- starts a 1.5 s jump
  slot at the next shared gait boundary (the base 0.75 s cycle and the 2.5x
  fast clock align every 2 base cycles, so one grid serves every command
  mode): the captured jump plays over its real 0.94 s inside the slot, the
  loop-closed landing pose holds the remainder, and the locomotion clock --
  running underneath -- exits the slot at its own boundary, so the transition
  needs no clock surgery and ports to the robot's wall-clock phase. A 0.06 s
  cross-fade at each slot edge makes the composite reference continuous by
  construction (raw boundary snaps of 0.2-0.5 rad drop below the references'
  own per-step motion). The commanded twist stays live through the slot --
  the policy learns to carry forward/backward/turn/sidestep velocity through
  the air from one identical jump reference -- and inside the slot the
  command blend is bypassed, so a stationary jump is still a jump. Rewards
  combine without gradient conflict via per-step time-disjoint masks keyed on
  the (policy-observable) slot: terms that fight a jump (``base_height``,
  ``lin_vel_z_l2``, ``air_time``, ``foot_slip``, the stand-still pair) switch
  off inside it, the jump height ladder (5/20, 0.40 m cap) switches on,
  linear velocity tracking drops only its vz^2 term in flight, one tracking
  term follows the composite reference with a loose in-slot std, and every
  regime-consistent term (yaw tracking, attitude, symmetry, torso clearance,
  action regularizers) stays on throughout. Same 48-dim frame and critic
  layout as MixedGaits, so those checkpoints warm-start it. The schedule is
  decided at reset and every consumer derives the slot purely from episode
  time, so the trigger is invisible to the policy until the reference swaps
  -- pre-jump anticipation cannot be learned. The play viewer gains a Jump
  button (the twist command's GUI, next to the joystick sliders) that
  schedules a slot at the next gait gap for the viewed env -- the
  interactive stand-in for the robot's eventual X button. After the first
  training round jumped in place regardless of command, the jump's active
  window was cut to the playback plus the exit fade (~1.0 s; slot starts
  stay on the 1.5 s grid): the v1 window held the landing crouch for the
  grid remainder and the policy learned exactly what that rewarded --
  killing its momentum to land stationary -- whereas now the reference
  stitches straight back into the flowing gait, the stride serving as the
  landing recovery. Velocity tracking also earns 2x inside the window, and
  the schedule densifies to one-to-three slots per episode (15% jump-free,
  from 25%). Two further dynamics passes: the reference window now ends a
  0.1 s lead *before* touchdown (~0.84 s into the capture), so the gait is
  feeding while the robot is still descending and the feet meet the floor
  already striding -- run, jump, run -- while the *reward* regimes keep the
  full-playback window (flipping the masked terms back mid-fall would
  punish ballistics), with the tracking tolerance loose through it. And the
  jump-start grid halves to one base cycle (0.75 s): with both edges
  cross-faded, exits land mid-cycle anyway, so starts no longer need the
  universal 1.5 s boundary -- trigger-to-jump lag halves (max 0.75 s), fast
  envs enter mid-cycle through the same fade (worst step 0.23 rad), and
  ``request_jump`` gains a busy guard so a trigger during an active window
  queues the next jump after it instead of restarting it mid-flight. The
  height ladder then doubles to 10/40 (task-local weights): the in-window
  velocity boost had raised the run-through payout to ~16/step, and at 5/20
  the policy monetized slots by simply keeping speed -- jump terms flat
  despite twice the slots, no jump at all under fast backward commands --
  whereas at 10/40 a full carried jump out-earns running through the
  window, making jump+carry the jointly dominant strategy. A companion
  ``Mjlab-MixedGaitsJump-Bumpy-Pupper-v3`` is the sim2real robustness
  finetune: the same task on the Perlin bump field (via ``_make_bumpy``,
  which also drops the critic height scan so flat checkpoints warm-start
  with identical shapes) with widened randomization -- foot friction
  0.35-1.6 (from 0.6-1.4), PD-gain scales kp 0.5-1.2 / kd 0.6-1.8 (from
  0.6-1.1 / 0.8-1.5) -- so the deployed policy has seen slicker floors and
  weaker motors than hardware will show it.
  ``pupper_mixed_gaits_env_cfg`` gains an optional ``base`` argument to
  build the mixed stack on the rough scene. The finetune also raises
  ``action_rate_l2`` to -1.0 (from the shared -0.01, which read ~-0.02/step
  against ~+12 of positives -- overwhelmed): a five-arm sweep of 300-iter
  finetunes from the nf1iv7oy checkpoint showed smoothing improving
  monotonically with weight (mean action delta 0.193 -> 0.169, 3-10 Hz
  action-power fraction 0.581 -> 0.475 -- the sim2real smoothness signal)
  with tracking, velocity carry, the jump terms and belly clearance flat at
  every weight tried, so the strongest arm ships. The finetune drops the
  airborne pretrain (it resumes a converged policy; pinning it airborne and
  zeroing rewards for 500 iterations would only delay the robustness
  objective).

- The Pupper deploy exporter handles the MixedGaitsJump tasks: their
  policies export as a new ``mixed_jump`` frame kind carrying the usual
  mixed ``gait_reference`` block plus a ``jump_slot`` block -- the captured
  jump table, its window geometry (``playback_s``, ``active_s``,
  ``cross_fade_s``, the 0.75 s scheduling grid and 1.0 s busy guard,
  asserted against the task constants at export) and the game-mode controls
  the deploy controller reads (``trigger_button`` 0 = X for the jump,
  ``run_button`` 1 = circle to hold for the fast gaits, and the
  ``walk_speed_cap`` 0.49 / ``run_speed_cap`` 1.5 m/s command caps). A
  numpy mirror of the composite reference
  (``mixed_jump_reference_offset_numpy``) backs a new ``slot_cases``
  section in the ``--golden-output`` parity fixture -- in-slot, cross-fade
  edge and slot-free cases cross-checked against the live training term
  before writing -- and the runner's automatic end-of-run export emits the
  same blocks.

- The Pupper deploy exporter handles the jump task: a ``Mjlab-Jump-Flat-
  Pupper-v3`` policy exports with a ``jump_reference`` block (the one-shot
  jump table, its trigger clock -- ``frequency``, ``crouch_hold_s``,
  ``phase_start`` -- and the joy ``trigger_button``) instead of
  ``gait_reference``, from the standalone exporter and the runner's automatic
  end-of-run export alike. ``--golden-output`` writes a jump parity fixture
  (cases keyed by seconds-since-trigger, including long-uptime landing-hold
  cases past anything a 1.2 s training episode can produce), cross-checked
  against the live training term before writing. The ``pupper_gait_deploy``
  controller plays the block one-shot on an R2 rising edge and holds the
  mid-stance crouch otherwise.

- Added ``Mjlab-Jump-Flat-Pupper-v3``: a single vertical jump per 1.2 s
  episode, maximizing apex height at a commanded twist pinned to zero. The
  reference (see ``mdp.jump``) holds the jump pronk's mid-stance crouch --
  phase 0, the touchdown keyframe with all feet forward, is statically
  unstable as a held pose -- for 0.3 s, plays exactly one cycle, and holds
  the same crouch for the landing. Height is driven by a dense two-term
  ladder: ``jump_up_velocity`` (upward base velocity clamped at zero -- the
  launch gradient, whose episode integral is proportional to total rise) and
  ``jump_airborne_height`` (base height above standing, gated on all four
  feet airborne so it cannot be farmed by standing tall). An exp-shaped
  ``base_height`` bump was tried first at weights 5 and 15 and converged to
  sitting in the crouch both times -- it pays nothing locally for pushing
  off the ground. The terms that fight a vertical jump at zero command are
  zeroed (``lin_vel_z_l2``, ``base_height``, the stand-still gates,
  ``pose``, ``air_time``), tracking uses the loose gallop tolerance so the
  policy can out-jump the modest one-shot reference, and the airborne
  pretrain warmup covers the first 300 iterations. The policy architecture
  is exactly the MixedGaits one (same 48-dim actor frame with the command
  dims at zero, same critic layout), so those checkpoints warm-start
  cleanly.

- Added a ``jump`` gait to the Pupper gait reference generator: the old
  IK-generated reach geometry (crouched to -0.11 m, 0.13/0.05 m stride, 3 cm
  lateral spread, 2.5x clock) with all four legs in phase -- a pronk -- plus a
  joint-space drive shaping that scales the hip's forward-side excursions by
  1 + ``_JUMP_HIP_FWD_GAIN`` (2.25x, clamped at the 2.51 rad hip limit) for a
  faster backward foot sweep through stance. Driven open-loop with gravity,
  syncing the legs raises the clean-hop apex above the standing height from
  0.11 m mean / 0.14 m p90 (staggered reach footfall) to 0.16 / 0.23, and the
  hip shaping doubles that again to 0.32 / 0.41. Amplifying the knee's
  backward excursions (``_JUMP_KNEE_BACK_GAIN``) hurt at every gain tried --
  the knee already saturates the 3 Nm actuator -- so it ships at 0, as does a
  roll tuck (``_JUMP_ROLL_TUCK``) that pulls the feet in under the body: it
  neither unloads the knee (saturation flat at ~46% of stance across the full
  tuck range, while the hip is the most-saturated motor at ~69%) nor raises
  the hop, because the lateral spread is what lets the crouched leg work
  straight. It is a separate gait name so the MixedGaits ``reach`` reference
  (now the captured table) is untouched; view it with
  ``visualize_reference --gait jump``.

- The Pupper deploy exporter (and the runner's automatic end-of-run export) now
  handles the mixed-reference gait tasks (MixedGaits, StableGait). Their deploy
  JSON ships the trot table, the reach table under the ``gallop_table`` fast
  slot (with the reach clock multiplier as ``gallop_freq_mult``), and the
  lift-in-place table as ``lift_table``, which the robot plays when the command
  is not translating. Previously these tasks failed the export outright (the
  frame-kind detection only knew the trot/gallop family), so runs ended with a
  warning and no ``policy.json`` on the W&B run. The exporter also stamps the
  ``heading_hold`` command contract into every deploy JSON, not just the one
  produced by ``export_pupper_gait_policy_from_env``, and the deploy parity
  fixture builder handles the mixed reference (and bypasses the heading hold so
  the fixture pins the raw command-to-offset mapping).

- The Pupper gait tasks close a heading-hold loop through the yaw command
  (``UniformVelocityCommandCfg.heading_hold_kp``), the same correction the G1
  carry tasks train with: walking with a quiet commanded yaw, the emitted yaw
  command becomes ``clip(kp * heading_error, +-0.3)`` toward the heading
  captured when the yaw went quiet, and a commanded turn or a stop disengages
  and re-arms it. This replaces the ``heading_deviation`` penalty (and its
  warmup curriculum): the penalty could not fix drift because neither actor nor
  critic can observe the held target, so it only added advantage noise, while
  closing the loop through the command works even on a frozen
  yaw-rate-tracking policy. The gait exporter stamps the parameters into a
  ``heading_hold`` block of the deploy JSON so the on-robot controller mirrors
  them off its IMU yaw; keep the two implementations in sync.

- Added ``Mjlab-StableGait-Flat-Pupper-v3``: the MixedGaits per-command
  selection without the fast reach branch. The triangular trot covers forward
  and backward (time-reversed) across the whole range, the lift-in-place cycle
  covers turning and sidestepping, commands cap at 0.7 m/s, and the phase clock
  runs quicker than the base cadence so the fixed trot stride slips less at
  the top of the range instead of switching gaits.

- Pupper training now writes the on-robot deploy JSON when a run ends -- both on a
  normal finish and on ``Ctrl+C`` -- and uploads it to the W&B run's *Files* as
  ``policy.json``, so the robot can fetch it directly. Deployment no longer needs a
  manual ``export-pupper-policy`` pass;
  ``pupper_gait_deploy/download_policy.py`` picks the file up unchanged.
  Implemented as ``PupperOnPolicyRunner``, so the shared runner and the G1/Go1
  tasks are untouched.

  On interrupt the in-memory policy is exported rather than the last checkpoint on
  disk: ``learn()`` skips its final save when interrupted, so the live policy is
  the freshest thing available. The upload uses ``policy="now"`` rather than
  W&B's default deferred sync, which would otherwise let a ``Ctrl+C`` tear the run
  down before the file was ever sent. Note that a hard kill or a cluster
  preemption signal does not raise ``KeyboardInterrupt``, so such a run leaves no
  JSON and still needs a manual export against its last checkpoint.

  The export parity-checks against the live actor and refuses to write rather than
  shipping a mismatched policy, and any failure is caught so it cannot mask a
  finished run. Observation normalizer statistics are unaffected: the algorithm is
  switched to eval mode for the export and restored afterwards, since
  ``get_inference_policy`` would otherwise leave it in eval for the rest of the
  run.

- Added ``Mjlab-MixedGaits-Flat-Pupper-v3``, which selects a gait reference per
  command instead of imposing one pattern on every motion: the original trot below
  the gallop onset, an extended-reach crouched gait above it in either direction,
  and a lift-in-place cycle for turning and sidestepping, where a fore-aft stride
  reference would oppose the commanded motion.

  The fast gait uses the gallop's footfall order rather than the trot's diagonal
  pairing, so no two legs move together -- a trot's paired legs cap how much ground
  a stride can cover, which is the wrong constraint for the fast mode. It runs a 2.5x
  phase clock (``_GAIT_FREQ_MULT``), lifting the reference's implied no-slip speed
  from 0.67 to 1.67 m/s so it no longer lags the 1.0 m/s top command. That costs
  headroom: peak demand goes to 4.81 Nm against a 3.0 Nm limit, so the reference is
  not literally trackable at this cadence -- 1.5x would give 1.00 m/s at 3.34 Nm,
  and 2.5x needs the reach cut to about 0.08 m to stay near the limit. Its swing
  lift is 0.08 m rather than 0.05: the planted foot must travel backward relative to the
  body to drive it forward, which the stance keyframes do, but a swing returning
  0.20 m forward on only 0.05 m of lift drags hard enough to cancel that. 0.08 cuts
  the travel-to-lift ratio from 4.0 to 2.5 and is the most the IK will take at this
  stance. A symmetric 0.05 m
  half-stride implies just 0.2 m/s of no-slip foot travel, so at speed the
  reference and the velocity command pull against each other; reaching 0.15 m
  forward against 0.05 m back implies 0.53 m/s. That reach is bought by crouching,
  not by loosening anything: the leg is 0.1734 m from the hip, so a foot planted
  0.14 m down has only sqrt(0.1734^2 - 0.14^2) = 0.102 m of horizontal travel left,
  and dropping the stance to 0.10 m raises it to 0.15 m. ``_GAIT_STANCE_Z`` and
  ``_GAIT_SWING_LIFT`` set depth and swing height per gait, the latter measured
  from the stance plane rather than absolutely -- against a fixed ``_SWING_Z`` a
  shallower stance would have silently eaten the foot clearance. Walking and
  turning keep the original 0.14 m stance, so only the fast mode crouches, which is
  also why ``base_height`` needs no special handling: it already relaxes above the
  same onset that selects it.

  ``_base_cycle`` now takes a swing-keyframe count and an asymmetric
  ``(front, back)`` stride. ``gait_reference`` also gains a ``cheetah`` entry from a
  kinematics study -- the reference ``gallop`` table has its footfall order
  inverted, since a cheetah rotary gallop lands the fore pair first, and its single
  swing keyframe pins the duty factor at 4/6 -- but no task uses it.

  Not deployable as-is: ``gait_reference.hpp`` on the robot mirrors the
  trot/gallop pair only, so the exporter would ship a reference block the
  controller cannot reproduce.

- Added ``Mjlab-MaxSpeed-Flat-Pupper-v3``, a study task for what gait emerges when
  the only objective is speed. Derived from the flat velocity task, which has no
  gait reference, with exactly two changes: commands are ``vx = +/-1.25`` m/s with
  zero lateral and yaw (``mdp.PupperMaxSpeedCommand``), and
  ``track_linear_velocity`` is scaled 5x to 7.5 so speed dominates, and its
  tolerance widened to ``std**2 = 0.5`` from the velocity task's 0.1. The widening
  is what makes the task trainable rather than a refinement: against a 1.25 m/s
  target the tracking reward is numerically zero below ~0.5 m/s at the original
  tolerance, so a robot that cannot yet run has no gradient at all and scaling the
  weight does not help -- 5x zero is still zero. At 0.5 the pull off standstill is
  70x stronger while the remaining incentive for the last stretch to the target is
  still worth about as much as any other single term.

  Every other reward, the domain randomization and the terminations are left
  exactly as the velocity task sets them, so an emergent gait difference is
  attributable to the objective rather than to a retune. A test pins that.

- The Pupper gait tasks now start with an airborne pretraining phase. For the
  first 500 PPO iterations (``GAIT_PRETRAIN_ITERS``) the base is pinned upright and
  stationary at ``GAIT_PRETRAIN_HEIGHT`` and every reward except ``gait_tracking``
  is zeroed, so the only gradient is "match the reference joint trajectory". The
  legs swing freely, contacts never fire, and balance is not yet a question; the
  pin then releases and the full objective comes back, with the policy already
  knowing the motion. The pin is a ``"step"``-mode event rather than a gravity
  change, since gravity lives in the compiled model and toggling it mid-run means
  recomputing derived constants. ``play`` skips both, having no curriculum manager
  to release the pin.

  ``mdp.pretrain_rewards`` must be registered as the *last* curriculum term: the
  drift and command ramps rewrite their own weights on every compute, so anything
  running after it would undo the zeroing mid-warmup. A test pins that ordering.

- ``track_angular_velocity`` returns to the Pupper gait tasks as a secondary
  roll/pitch-rate stabilizer at weight ``1.0``, alongside -- not replacing --
  ``track_yaw_velocity`` at ``3.0``. Folding the squared roll/pitch rate into the
  yaw error is what made it unusable as *the* yaw signal, since body wobble
  consumed the budget and left yaw no gradient. As a secondary term against a clean
  yaw signal that same contamination is the point: what it contributes is damping.

- Added ``Mjlab-TrotGallop-Flat-Pupper-v3`` and
  ``Mjlab-TrotGallop-Bumpy-Pupper-v3``, which trot up to 0.5 m/s and gallop from
  0.5 to 1.0, with the command range extended to cover the gallop band. Separate
  task ids from the trot-only tasks, which keep the gallop gated off, so trot
  results stay comparable across runs. Note that re-enabling the gallop does not
  make it trackable: the reference still demands 1.44 rad on a knee within one
  20 ms control step, needing 7.9 Nm against a 3.0 Nm effort limit, and still has
  no suspension phase. Measured with random actions, ``gait_tracking`` scores
  0.22 raw in the trot band against 0.09 in the gallop band. Pass
  ``std=GALLOP_STD`` to keep more gradient above the switch, at the cost of
  loosening the trot band too.

- Added ``Mjlab-Trot-Flat-SharedCritic-Pupper-v3``, an ablation task with a
  symmetric actor-critic: both networks receive the same 48-dim proprioceptive
  frame, fully corrupted (per-subgroup noise plus IMU latency), with every
  privileged critic term dropped. This mirrors ``pupperv3-mjx``, whose
  ``environment._get_obs`` returns a single flat array handed to both networks
  with no ``value_obs_key`` -- it has no privileged critic observation, and no
  un-randomized variant of one. The task also skips the yaw curriculum, opening
  the full +/-2 rad/s range from step 0 while keeping the heading-deviation
  curriculum. Because the IMU latency buffer is owned by the env and shared, the
  frame is computed once per step and cached (``mdp.pupper_gait_shared_obs``);
  letting both groups build it independently would advance the buffer twice per
  step and halve the effective lag.

- The Pupper critic now observes the *applied* motor target
  (``mdp.applied_action``) alongside the policy's raw output. The command lag is a
  per-step, per-env coin flip, so the two disagree on ~80% of steps and nothing in
  the observation said which had happened -- leaving the value function to predict
  returns from a transition whose input it could not see, with the unexplained
  variance landing in the advantages. The pair also determines the realized lag.
  This is privileged and critic-only: the robot cannot observe its own bus lag, so
  the actor keeps the unchanged 48-dim deploy frame and the export is unaffected.
  Note the critic input grows 84 -> 96, so Pupper checkpoints from before this
  change will not load; retrain or warm-start the actor only.

- Added the ``export-pupper-policy`` script, which converts a trained Pupper
  policy into the on-robot ``neural_controller`` deploy JSON, parity-checks it
  against the source actor on real observations, and can upload it to the W&B run
  for the robot to fetch. Gait ("mimic") policies additionally get a
  ``gait_reference`` block carrying the phase reference tables, since the robot
  has to reproduce the motion reference the policy observes; pass
  ``--golden-output`` to also emit the fixture the robot-side parity test uses.

Changed
^^^^^^^

- The jump task's reference is now a *captured* jump rather than the shaped IK
  pronk, following the MixedGaits playbook: run ``zuy6c85c``'s emergent
  countermovement, recorded from the loading dip through the landing across 8
  phase-locked episodes (liftoff at 0.56 s in every one), left/right
  mirror-averaged (1.58 rad of asymmetry removed -- the policy jumped visibly
  lopsided) and loop-closed onto its own near-stance hold pose. One playback
  spans the capture's real 0.94 s (``JUMP_FREQUENCY`` becomes 1/duration;
  phase starts at 0), and open-loop it launches to 0.55 m from a standstill --
  the windup is baked in as planted motion. The first hardware test also
  exposed a sim2real exploit this rework closes: the policy bounced a windup
  off the reset drop's rebound (the robot spawned 4.5-8.5 cm in the air),
  energy a grounded robot does not have. The jump task now spawns settled at
  standing height with no z randomization, holds the crouch 0.6 s (from 0.3)
  so reset transients die before launch, gates both jump rewards to the
  post-launch window so pre-hop bouncing earns nothing, and keeps the episode
  running 1.2 s past the playback (2.74 s total) -- the reference holds the
  crouch there and only a stuck, settled landing keeps earning the tracking
  and zero-velocity terms, so landing recovery is trained rather than
  surviving to the buzzer. The jump task also adopts MixedGaits'
  ``roll_asymmetry`` penalty (same -30.0 weight): the symmetrized reference
  shows the policy the template, and the penalty makes drifting back into
  the lean expensive. Tracking tolerance is now phase-dependent: strict
  (trot std) during the pre-launch hold and the settled recovery, loose
  (gallop std) only inside the playback window plus a 0.4 s landing grace.
  Run ``e9194f0a`` exposed why one loose std is not enough: the reward gate
  stops hold-phase *motion* from earning, but the policy parked in a splayed
  crab crouch during the hold -- a *pose* -- and converted that geometry
  into launch energy when the gate opened; only tight tracking prices the
  pose. The height ladder also steps back down to its original 5/20 weights
  with the airborne term capped at 0.40 m above stand: 15/60 existed to
  *discover* jumping against the feeble IK reference, and with the captured
  reference that job is done -- at 15/60 the height terms paid 8.9/step and
  climbing against ~3.1/step of falling form terms (run ``f24c7x9l``,
  descending in a contorted pose), while past the cap a centimeter now pays
  nothing and marginal capacity goes to tracking, symmetry and attitude.
  A ``jump_hold_descent`` penalty (-10.0 on downward base speed, active only
  during the pre-launch hold) closes the last momentum route: the reward
  gate keeps hold motion from earning and the strict std prices the pose,
  but dropping the body through tolerable poses could still bank launch
  momentum -- now it costs more than the capped height terms can repay,
  while the countermovement dip inside the playback stays free. The
  airborne pretrain warmup stays (it was briefly removed on the theory that
  the captured reference is trackable from the ground anyway, but without
  it the policy struggles to memorize the motion with everything else
  live): the reference input during the hold and recovery phases is the
  static crouch pose, so a pinned policy spends most of pretrain practicing
  exactly the static mapping under the strict hold-phase std, and every
  reward but tracking is zeroed while pinned, so the pin can neither farm
  the height terms nor be taxed by the descent penalty. A
  ``torso_clearance`` penalty (-8.0) prices the torso's lowest point
  entering a 2 cm floor no-hit zone (orientation-aware, from the base
  body's compiled bounding box -- the belly sits 4.6 cm under the base
  origin): the torso has no collision geom, so sim charged nothing for the
  belly passing through the ground, the zuy6c85c-lineage policies loaded
  2+ cm below the floor every episode, and on hardware that is the battery
  slamming the ground and disconnecting the robot. Graded rather than the
  termination it briefly was -- a hard wall priced a 1.9 cm dip like a
  belly slam and cost too much jump height, while the 0-to-1 ramp across
  the zone (past 1 below the floor) lets the load shave in cheaply and
  makes ground contact expensive (~11/step at the old below-floor depth).
  Delete ``captured_jump_gait.npz`` to fall back to the IK pronk.

- The MixedGaits fast references are now *captured* gaits rather than
  IK-generated ones: the emergent gaits run ``al7sdood`` actually performed at
  +1.5 and -1.5 m/s, recorded phase-locked to the reference clock
  (cycle-to-cycle dispersion under 6 mrad at 98-100% of envs on-speed) and
  shipped as ``captured_reach_gaits.npz``. Forward and backward are separate
  captures (``reach`` / ``reach_back``) selected by command sign -- how the
  robot moves backward fast is not the forward gait time-reversed. Each
  left/right leg pair of the captures is mirror-averaged (all three joints,
  one shared phase shift) before shipping: the raw policy's worst asymmetry
  was the leg roll -- motor 2, the joint that moves the foot in/out, as a
  FK perturbation check confirms -- with pair sums of 0.44-0.68 rad held for
  the whole gait, i.e. one leg tucked in while its partner splayed out. A new
  ``roll_asymmetry`` penalty (EMA of the left+right motor-2 pair sums,
  squared, weighted ``-30.0``) makes sustaining such a lean expensive during
  training while ignoring the legitimate within-cycle roll oscillation.
  ``gait_tracking`` rises to ``5.0``
  (``MIXED_GAIT_TRACKING_WEIGHT``): the reference now *is* the fast behavior,
  so tracking it no longer fights the velocity objective. The exporter ships
  the backward capture as an optional ``gallop_back_table`` (pre-reversed, so
  the controller's shared backward phase reversal plays it forward as
  recorded), and the ``pupper_gait_deploy`` controller selects it for
  backward-fast commands -- older policies without the key keep the previous
  time-reverse-the-single-table behavior.

- The MixedGaits task weights ``track_linear_velocity`` at ``8.0``
  (``MIXED_TRACK_LIN_VEL_WEIGHT``) rather than the ``3.0`` the other gait
  tasks share: the per-command references exist to chase speed, so velocity
  outweighs ``gait_tracking`` decisively against the widened +-1.5 m/s
  commands. 8.0 was previously tried on top of the raised-rear recipe that
  got reverted wholesale; with the spread reach geometry and the 45 deg
  termination it gets a clean second run.

- The MixedGaits task widens its linear velocity commands to +-1.5 m/s
  (``MIXED_MAX_LIN_VEL_X``) and loosens the fall-over termination to 45 deg
  (``MIXED_FELL_OVER_LIMIT_ANGLE``). The 30 deg tilt limit -- the notebook's
  ``terminal_body_angle``, tuned for the trot -- cut episodes exactly when the
  policy committed to the fast reach gait at speed, since a fast gait
  legitimately pitches harder than a trot; 45 deg still catches real falls
  (the base velocity task uses 70). With the spread/raised reach geometry
  implying ~1.2 m/s of no-slip travel at the 2.5x clock, 1.5 m/s stretches
  the reference about as far as the old 1.0 m/s command stretched the old
  one. MixedGaits only; the TrotGallop tasks keep the shared 1.0 m/s range
  and 30 deg limit.

- The MixedGaits reach gait plants its feet 3 cm outside the hip line
  (``_REACH_SPREAD``), crouches 1 cm less (stance ``-0.10 -> -0.11``) and
  reaches 2 cm shorter (``0.15 -> 0.13`` m forward). The abduction axis is
  along x, so the spread tilts the leg planes outward without touching the
  fore-aft stride geometry -- the leg works straighter instead of deeply
  folded, which is what the uniform crouch was demanding. Swept jointly, the
  trio beats the shipped geometry on every measured axis: max IK residual
  2.0 -> 1.27 mm, peak per-step actuator demand 5.02 -> 4.09 Nm against the
  3.0 Nm limit, mean stance knee angle 0.451 -> 0.398 rad, and max abduction
  1.58 -> 1.40 rad. Spread alone at the old geometry is infeasible (residual
  8.7 mm) -- the workspace was already spent. Costs ~10% of implied no-slip
  speed (0.18 vs 0.20 m of stride per cycle). StableGait is unaffected: its
  reach branch is gated off. Re-export any MixedGaits policy trained before
  this change from its own checkout, since the deploy tables must match what
  the run trained on.

- The StableGait phase clock drops from 2.0 Hz to 1.5 Hz
  (``STABLE_GAIT_FREQUENCY``): at 2.0 Hz (runs ``g6ugoiq2`` and ``1mc78ipo``)
  backward, turning and sidestepping were smooth but forward walking was not,
  and at 50 Hz control a 2.0 Hz cycle leaves only ~7 control steps per swing to
  place a foot. 1.5 Hz gives ~11 while keeping 1.125x the base cadence's
  no-slip travel.

- Pupper training now exports and uploads the deploy ``policy.json`` at every
  checkpoint save (each ``save_interval`` iterations plus the final save),
  overwriting the previous upload under the stable name, instead of only when
  the run ends. The W&B run therefore always carries a deployable JSON of its
  newest checkpointed policy, so a mid-run policy can be put on the robot
  without a manual export, and a hard kill or preemption at worst loses the
  iterations since the last save. The parity check that gates each export is
  capped at 512 environments to keep the per-checkpoint cost negligible; the
  Ctrl+C path still exports the in-memory policy, which is fresher than the
  last on-disk checkpoint.

- The StableGait task weights ``foot_slip`` at ``-0.2``
  (``STABLE_GAIT_FOOT_SLIP_WEIGHT``), 2x the shared reward set's ``-0.1``: the
  ``g6ugoiq2`` policy walked backward, turned and sidestepped smoothly but was
  rough going forward, where the fixed trot stride slips most against the top
  of the 0.7 m/s command range. First raised to ``-1.0`` (run ``1mc78ipo`` and
  the first 1.5 Hz run), but at that strength it overshadowed the action-rate
  penalty, so it comes back down to a modest bump. StableGait only; every
  other Pupper task keeps the answer-key weight.

- The Pupper gait tasks raise all three headline positive terms to ``3.0``:
  ``gait_tracking`` ``1.5 -> 3.0``, ``track_linear_velocity`` ``1.5 -> 3.0``, and
  ``track_yaw_velocity`` ``0.8 -> 3.0``. The gait was plateauing well short of the
  reference and the velocity signals were too weak against everything else; yaw
  gets more than a doubling because it was the slowest to climb. Equal weights keep
  gait, forward speed and turn rate carrying equal say rather than one taking over.
  Set in the gait config, so the plain velocity tasks keep the recovered
  ``pupperv3-mjx`` weights (1.5 / 0.8) as the sim2real baseline. Note this scales
  the positives against unchanged regularizers, so if the gait gets loose or jerky
  the action-rate and attitude penalties are the first things to rebalance.

- The Pupper gait reward now grades against the reference rewound by the pipeline
  delay (``GAIT_REWARD_PHASE_LEAD_STEPS``, 1.8 control steps: one from
  ``episode_length_buf`` incrementing before the reward, plus 0.8 for the
  actuation lag). Previously the policy observed ``ref(k)`` and was graded against
  ``ref(k+1)`` with its action landing later still, so it had to learn a constant
  feedforward lead purely to break even -- a policy whose joints lag by exactly
  the pipeline delay scored 0.33 instead of 1.00. This is training-signal only:
  the *observation* keeps a zero lead, since the on-robot controller has to
  reproduce it exactly, and a constant phase shift of a periodic gait is
  unobservable anyway. Note it raises the absolute value of ``gait_tracking`` for
  identical behavior, so reward curves are not comparable across this change.

- The Pupper tasks no longer apply base-velocity kicks (``push_robot``). The
  reference pipeline does kick, but far more gently -- 0.2 m/s at 2% per step,
  linear only -- whereas the inherited velocity-task event applies +/-0.5 m/s plus
  z and full angular kicks every 1-3 s, which the reference never does. Disabled
  rather than retuned; a linear-only 0.2 m/s kick can be re-added if the policy
  proves fragile to shoves on hardware.

- The Pupper command latency is now modelled as a *deterministic* 4-physics-step
  (16 ms) actuator delay instead of the reference pipeline's per-control-step
  ``[0.2, 0.8]`` Bernoulli. The robot's bus delay is fixed -- the deploy stack
  reports ``command_latency_timesteps: 8`` at 520 Hz, i.e. 15.4 ms -- which is 0.77
  of a 20 ms control step. ``[0.2, 0.8]`` is how ``pupperv3-mjx`` expresses a
  sub-step delay given only a whole-control-step buffer; its *mean* of 0.8 steps
  is the quantity that matches hardware, and the per-step coin flip is a
  discretization artifact rather than jitter the robot has. mjlab integrates at
  4 ms and can hold the same 16 ms exactly. This matters because the realized lag
  was invisible to the policy (``last_action`` is its own raw output), so it could
  never learn the correct phase lead and hedged between both branches instead --
  swamped by the reference while walking, dominant at low authority, where it
  showed up as residual stutter in both sim and hardware.

  ``PUPPER_LATENCY_MODEL`` selects between ``"deterministic"`` (new default) and
  ``"mjx"`` (bit-for-bit reproduction of the reference), so the answer-key path
  stays available for comparison. Under the deterministic model the critic drops
  ``applied_action``: with a constant lag the applied target is a fixed function
  of the policy's own history, so there is no hidden variable to expose, and the
  critic input returns to 84 dims.

- The Pupper gait tasks now ramp three penalties to 3x their base weight over
  training, against long-horizon drift seen on hardware: ``heading_deviation``
  ``-2.0 -> -6.0``, ``orientation_l2`` ``-5.0 -> -15.0``, and ``ang_vel_xy_l2``
  ``-0.05 -> -0.15``. The ramp is two-stage -- base, then 2x at 10,000 PPO
  iterations, then 3x at 15,000 -- because at full strength from step 0 these
  compete with linear and yaw velocity tracking before either is established, and
  because a single jump to 3x is a large mid-training shift for a converged
  policy. Tasks using the heading warmup hold that term at zero and go straight to
  3x at the same 15,000, so both paths end at the same weights and stay
  comparable. ``play`` gets the final weights directly. The two attitude weights
  are set in the gait config rather than the shared Pupper reward set, so the plain
  velocity tasks keep the recovered ``pupperv3-mjx`` values as the sim2real
  baseline.

  Weight is only half of the heading story, though: ``heading_hold_target`` is
  recaptured on every command resample, and ``resampling_time_range`` is
  ``(3.0, 8.0)`` against a 10 s episode, so nothing penalizes drift accumulated
  beyond ~8 s at any weight. ``pupperv3-mjx`` resamples once per 10 s episode
  (``resample_velocity_step=500``), effectively holding one command for the whole
  episode. Aligning that is left as a separate change so the two effects stay
  attributable.

- The Pupper trot/gallop tasks now run without either curriculum: the full
  ``|yaw| <= 2`` rad/s command range and the ``heading_deviation`` penalty are both
  live from step 0. Both curricula existed to work around yaw tracking that would
  not climb, which turned out to be the shared velocity term folding roll/pitch
  rate into the yaw error rather than anything about the command schedule. With
  ``mdp.track_yaw_velocity`` in place the scaffolding is worth removing -- it is
  also what unblocked the gallop, since a gallop wobbles far more than a trot and
  the old term taxed it hardest. ``_add_gait`` gains ``heading_warmup``; the
  trot-only and gallop-only tasks keep both curricula. Task registration now reads
  the curriculum off the env config instead of assuming every gait task has one,
  so tasks without it no longer draw a spurious reachability warning.

- The Pupper tasks now track yaw with ``mdp.track_yaw_velocity``, a direct port of
  ``pupperv3_mjx.rewards.reward_tracking_ang_vel``, replacing the shared velocity
  term. That term adds the squared roll/pitch rate to the yaw error, so trot body
  wobble consumed most of the reward: measured mid-trot it was 59% of the total
  error, and even *perfect* yaw tracking capped the term at 0.12, against 0.35
  under the reference form for merely mediocre yaw. Yaw tracking had almost no
  gradient to climb. The reference divides by ``sigma`` directly rather than
  ``std**2``, so the parameter is named ``sigma`` (0.25, as upstream) and the
  weight returns to the reference's 0.8 from 2.0. Roll/pitch is still penalized by
  ``ang_vel_xy_l2``, exactly as upstream, so this removes a double penalty rather
  than a check. The term is renamed rather than re-parameterized in place, since
  it measures a different quantity and curves from before and after are not
  comparable. Only the Pupper tasks change; G1 and Go1 keep the shared term.

- The Pupper gait tasks now ramp the commanded yaw-rate range instead of offering
  the full |yaw| <= 2 rad/s from the first step: 1.0 until 5,000 PPO iterations,
  1.5 until 10,000, then 2.0 (``GAIT_YAW_STAGES``). Yaw tracking was learning, but
  slowly, against the full range from scratch. This re-adds a ``command_vel``
  curriculum that the gait tasks previously dropped outright; the inherited one
  ramped every axis and held the linear commands back, whereas these stages touch
  yaw only. ``play`` gets the final range directly, since it has no curriculum
  manager and would otherwise evaluate at the stage-0 range forever.

- The Pupper gait tasks now hold the ``heading_deviation`` penalty at zero until
  15,000 PPO iterations (``GAIT_HEADING_WARMUP_ITERS``), then switch it to its
  full ``-2.0``. The penalty is gated on ``|yaw_cmd| < GAIT_HEADING_YAW_MAX`` so it
  never fires on a commanded turn, but it still teaches "do not yaw", which
  suppresses the yaw response and stalls ``track_angular_velocity``. The tasks'
  ``max_iterations`` default rises 10,000 -> 20,000 to match; with the previous
  default the run ended before the penalty switched on, making the curriculum a
  silent no-op. Task registration now warns if ``max_iterations`` is ever set
  below the threshold, or if ``num_steps_per_env`` changes such that the
  iteration-to-step conversion no longer holds.

- Added ``Mjlab-Gallop-Flat-Pupper-v3``, a dedicated gallop task on flat ground
  with forward-only commands over 0-1 m/s. The gallop reference replaces the trot
  outright rather than switching in above a speed threshold, and
  ``gait_tracking``'s tolerance is loosened 2x (``GALLOP_STD``) so the reference
  acts as a shape prior while velocity tracking drives the behavior -- the
  reference is not currently trackable, so a tight tolerance would saturate the
  reward near zero and leave no gradient. It logs to its own ``pupper_gallop``
  experiment. Expect this task not to train until the reference is reworked; it
  exists so that work can proceed in isolation.

- The Pupper trot tasks' maximum forward/backward command drops from 1.0 to
  0.7 m/s (``GAIT_MAX_LIN_VEL_X``). The 1.0 existed only so the gallop could cover
  the top of the range; with the gallop split into its own task the trot no longer
  needs it, and the trot reference's foot travel only implies ~0.2 m/s anyway.

- The Pupper gait tasks' gallop reference is disabled: ``GAIT_GALLOP_SPEED`` is
  now above the maximum command, so a single trot reference covers the whole speed
  range. The gallop was not trackable by the robot -- it demanded 1.44 rad on a
  knee within one 20 ms control step, needing 7.9 Nm against a 3.0 Nm effort limit
  (the trot's worst case is 0.37 rad / 2.05 Nm), which is why trained policies
  imitate the trot cleanly and fail the gallop. Sweeping stride against cadence
  found nothing both actuator-feasible and faster than ~0.25 m/s, below the
  0.5 m/s that used to select it. Despite its name it also had no suspension
  phase: ``_base_cycle`` fixes the duty factor at 4/6, so at least two feet were
  always down, exactly like the trot. The tables and parameters remain wired up,
  so re-enabling is a one-line change once a trackable reference exists. The
  exported deploy JSON carries the same threshold, so the on-robot controller
  gates identically.

- The Pupper tasks now model the deploy stack's control latency, closing a
  sim2real gap that made trained policies oscillate on hardware while walking
  cleanly in sim. The robot lags in both directions -- its IMU reading is up to
  one control step stale (~19 ms) and the motor target arrives up to one step
  late (~15 ms) -- and neither was simulated. Injecting them reproduces the
  measured hardware behavior closely (body angular velocity 0.12 -> 1.16 rad/s
  against 1.28 on the robot; 3-10 Hz action power 22% -> 43% against 44%), and
  they are super-additive: each alone accounts for barely a third of the effect,
  which is why neither was caught in isolation. The distributions match the
  ``pupperv3-mjx`` pipeline the tasks are ported from
  (``mdp.PUPPER_ACTION_LATENCY_DIST``, ``mdp.PUPPER_IMU_LATENCY_DIST``). Lag
  reaches only what it reaches on the robot: the 6 IMU dims (post-noise), and the
  applied motor target -- never ``last_action``, which is the policy's own output,
  nor the joint encoders, which are read straight through.

- Pupper ``play`` configs now keep observation noise and latency enabled, matching
  the MJX pipeline's eval env, which is built with the same parameters as
  training. These model permanent properties of the robot rather than training
  regularizers, and the latency especially cannot be dropped at eval: its median
  lag is nonzero, so serving undelayed observations yields a materially different
  plant. A latency-free play mode is what made the sim look healthy while the
  robot shook.

- The Pupper deploy JSON now records ``single_observation_size`` so the on-robot
  controller can size its observation frame from the policy rather than a
  compile-time constant. Gait policies use a 48-dim frame (36 proprio + the
  12-dim reference offset they track) instead of 36.

- The Viser reward bar panel's term cap is now configurable via
  ``ViewerConfig.reward_bar_max_terms``, so environments with more than 20
  reward terms can show them all. Defaults to 20, preserving previous behavior.
  :issue:`1079`

Added
^^^^^

- The Pupper tasks now penalize foot-ball self-collision (``self_collision_r`` and
  ``self_collision_l``, each at ``-10.0``, matching the knee-on-ground penalty). A
  foot-to-foot strike is rare but wrecks the stride on hardware, and nothing saw
  it: the model simulates these contacts, every leg geom being contype/conaffinity
  1 with no exclusion pairs, but no sensor or reward covered them.

  One sensor per side, front foot against back foot. Two details make the obvious
  spelling silently useless. ``ContactSensor`` resolves ``secondary`` to a *single*
  element, so a pattern matching both back feet watches only one of them and the
  other pair goes unseen. And sampling reachable poses inside the real joint
  limits, only the same-side front/back pairs can meet at all -- ``front_l``/
  ``back_l`` closing to -0.030 m and ``front_r``/``back_r`` to -0.024 m -- while
  every left-right pair stays at least 0.050 m apart because abduction runs out of
  travel first, so a left-versus-right sensor cannot fire at any pose.

  Verified end to end: at a pose where the two left foot geoms overlap, the sensor
  reports contact and the reward reads -10.0 while the right-side term stays at 0;
  at the default pose both read 0.

Fixed
^^^^^

- Fixed the Viser play viewer crashing on tasks whose velocity command range
  is pinned to zero (e.g. the Pupper jump task): the joystick GUI's "Max"
  slider was created with the range's upper bound as its initial value, below
  the slider's own 0.1 minimum, tripping viser's bounds assert. The axis
  limit is now floored at 0.1; the joystick only writes to envs when enabled,
  so the small manual range on a pinned axis is harmless.

- Fixed the per-checkpoint ``policy.json`` upload not replacing the file on a
  *live* W&B run: ``wandb.save(policy="now")`` uploads a registered path once,
  so mid-run the Files tab kept the first checkpoint's export until the
  end-of-run sync (and the Files tab's ``updatedAt`` shows the first upload
  time even after a replacement, which is what made the staleness hard to
  see). The runner now pushes each export through
  ``wandb.Api().run(...).upload_file``, which overwrites the server-side file
  synchronously on every save.

- Fixed every save-time W&B upload silently doing nothing: the runners gated
  uploads on ``self.logger.logger_type == "wandb"``, but rsl_rl's ``Logger``
  rewrites that attribute to ``"WandbLogWriter"`` during init, so the ONNX and
  Pupper deploy-JSON exports succeeded locally and then skipped ``wandb.save``
  without a warning (checkpoints still arrived, uploaded by wandb's own file
  sync). The gates now go through ``wandb_logging_active``, which accepts both
  spellings -- the tracking runner already did this. Runs trained before the
  fix have the deploy JSON in their local log dir but not on the run; upload
  it manually or re-export with ``export-pupper-policy``.

- On Linux, mjlab now probes for a usable EGL device before defaulting
  ``MUJOCO_GL=egl``. Previously ``import mujoco`` crashed at import time on
  machines without an EGL userland (e.g. CUDA compute-only cluster
  containers), even for headless training that never renders through GL.
  When EGL is unavailable and ``MUJOCO_GL`` is not set explicitly, mjlab
  falls back to ``osmesa`` (if present) or ``disabled`` with a warning; an
  explicit ``MUJOCO_GL`` still always wins.

- Fixed the Pupper gait reference tables being clamped to the wrong joint limits.
  ``_joint_limits`` looked up bare joint names, but the compiled scene prefixes
  every element with the entity name (``robot/leg_front_r_1``), so every lookup
  returned ``-1`` -- and ``jnt_range[-1]`` is a valid row rather than an error, so
  every joint silently received the *last* joint's range, ``[-0.710, +2.790]``,
  instead of its own. Limits now come from the entity's own indexing, the same fix
  the exporter already had.

  It was wrong in both directions. On the forward-drive joints the real range is
  ``[-1.220, +2.510]``, so 0.51 rad of travel was discarded; on the knees the real
  upper limit is ``+0.710`` against the ``+2.790`` being applied, so the reference
  could ask for angles the hardware cannot reach. 0.6% of the trot table and 4.2%
  of the extended-reach table exceeded the real limits, peaking at 0.830 rad on
  ``leg_front_r_3`` against its 0.710 limit -- silently truncated by the on-robot
  controller, which is a sim/real divergence rather than a visible failure.

- Fixed the Pupper actor observation leaving its noised projected-gravity vector
  unnormalized. The robot derives projected gravity from the IMU quaternion, so it
  is always a unit direction, but training perturbed the vector without
  renormalizing and so fed the policy a ~5% norm variation the hardware never
  produces. Now renormalized after noise, matching the ``pupperv3-mjx`` env.

- Fixed ``export_pupper_policy_from_env`` looking up joint limits and actuator
  gains by bare joint name. The compiled scene prefixes elements with the entity
  name (``robot/leg_front_r_1``), so every lookup returned id ``-1`` and the
  export raised ``IndexError`` instead of producing a deploy JSON. Limits now come
  from the robot entity's own data and gains from its actuator indices.
- The Viser reward bar panel no longer *silently* drops reward terms beyond
  ``max_terms``; it now emits a warning listing the hidden terms. Previously
  environments with more than 20 reward terms had the overflow disappear from
  the bar panel with no indication. :issue:`1079`
- Fixed the ``terrain_levels_vel`` curriculum promoting every env from level 0
  to level 1 on the initial reset, ignoring ``max_init_terrain_level=0``. Before
  the first step the robot sits at its spawn pose rather than a walked-to
  position, so the distance check was spurious; terrain levels are now frozen on
  that first reset. :issue:`1094`
- Fixed the velocity task's actor ``joint_pos`` observation not being biased by
  the ``encoder_bias`` domain randomization, so the encoder bias only affected
  actions and never the observed joint positions. The actor now observes biased
  joint positions while the critic keeps the true (unbiased) values as
  privileged information, matching the tracking task.
  See `discussion #1065 <https://github.com/mujocolab/mjlab/discussions/1065>`_.
- Hardened ``fit_terrain_normal`` against non-finite raycast hits. A single env
  with a diverged state produced a NaN/Inf covariance that made
  ``torch.linalg.eigh`` raise and abort the whole batch; such rows now fall back
  to the up vector. This stops the hard crash so a diverged env can be reset
  normally; it does not by itself make a diverged env's downstream reward finite.
  :issue:`912`
- Enabled ``obs_normalization`` on the Go1 velocity actor and critic to match
  the other velocity tasks. Without it, extreme-but-finite observations on rough
  terrain drove value/policy divergence that eventually surfaced as a
  ``normal expects all elements of std >= 0.0`` crash. Note that Go1 velocity
  checkpoints trained before this change carry no normalizer buffers and will no
  longer load; retrain from scratch. :issue:`870` :issue:`1044` :issue:`1053`
- Fixed ``ContactSensor`` air-time tracking accumulating float32 sim-clock
  differences, whose quantization error grows with the clock magnitude and made
  ``compute_first_contact`` / ``compute_first_air`` miss touchdowns on long runs.
  The exact float64 substep ``dt`` is now accumulated instead. :issue:`1101`

Version 1.5.2 (July 17, 2026)
-----------------------------

Fixed
^^^^^

- Fixed CUDA illegal memory accesses when domain randomization triggers
  ``set_const`` with multiple environments. ``actuator_acc0`` is now expanded
  per environment before MuJoCo Warp recomputes it.
- Fixed ``MaterialCfg.reflectance`` being ignored when building the MuJoCo
  spec. Contribution by @bd-pmorais.

Version 1.5.1 (July 15, 2026)
-----------------------------

Added
^^^^^

- Added ``MeshCfg``, a spec editor that matches mesh assets by name and edits
  their asset-level attributes. The first attribute is ``maxhullvert``, which
  caps the collision convex hull's vertex count to lower narrowphase cost.
- Added ``SimulationCfg.broadphase`` and ``SimulationCfg.broadphase_filter``
  to configure MuJoCo Warp's broadphase collision algorithm and
  bounding-volume filters.

Changed
^^^^^^^

- Enabled skybox rendering for camera sensors. Contribution by @bd-pmorais.
- Bumped the minimum ``mujoco-warp`` to 3.10.0.2, which fixes ``qfrc_constraint``
  being populated incorrectly across vectorized environments (:issue:`1086`).
  Earlier 3.10.0.x releases are no longer supported.
- Command delay on fusable actuators (ideal PD, DC motor) now applies one shared
  lag per environment across all fused actuators sharing a delay config, matching
  the built-in actuator path, rather than an independent lag per actuator group
  (:issue:`1035`).

Fixed
^^^^^

- Fixed ``TerrainGenerator`` overwriting custom geom names set by sub-terrain
  functions with the default ``terrain_{i}`` name. Only unnamed geoms are now
  auto-named.
- Fixed ``TorchArray`` not expanding world-shared model fields to ``nworld``
  with mujoco_warp 3.10.0.2, which allocates them as real size-1 arrays
  instead of stride-0 broadcast views. Multi-env indexing of fields like
  ``soft_joint_pos_limits`` raised ``IndexError`` during resets (:issue:`1093`).
- Fixed ``mdp.bad_orientation`` returning NaN when float32 rounding in
  ``quat_apply_inverse`` pushed the projected-gravity z-component slightly
  outside ``[-1, 1]``, making ``torch.acos`` return NaN and silently
  suppressing the termination for flipped robots. The argument is now clamped
  to ``[-1, 1]``.
- Fixed a crash when using command delay on ideal PD (or other custom)
  actuators whenever ``num_envs`` differed from the number of delayed targets,
  and fused ideal PD and DC motor actuators sharing a transmission and delay
  config into a single gather, delay, control-law evaluation, and control
  write, removing per-group host overhead (:issue:`1035`).

Version 1.5.0 (June 28, 2026)
-----------------------------

Added
^^^^^

- Added ``reduce="max"`` to ``MetricsTermCfg`` for reporting episode-peak values
  (e.g. peak power, peak contact force) without needing stateful wrapper classes.
- Added ``BuiltinDcMotorActuator``, a native MuJoCo ``<dcmotor>`` wrapper.
  Supports voltage / position / velocity input modes with back-EMF,
  configurable motor constants, and optional integral, slew, inductance,
  thermal, LuGre, and cogging extensions.
- Added ``scale_with_difficulty`` to ``HfRandomUniformTerrainCfg``. When
  enabled, the noise amplitude scales with difficulty (flat at 0, full
  ``noise_range`` at 1) so the terrain progresses in a curriculum. Defaults to
  ``False``, preserving the previous difficulty-independent behavior.
- Added material domain randomization functions for MuJoCo Warp RGB rendering:
  ``dr.mat_emission``, ``dr.mat_specular``, ``dr.mat_shininess``, and
  ``dr.mat_texrepeat``.

Changed
^^^^^^^

- Bumped ``rsl-rl-lib`` from 5.2.0 to 5.4.0.
- Bumped ``mujoco`` and ``mujoco-warp`` to 3.10, both pinned from PyPI. The
  ``py.mujoco.org`` nightly index and the ``mujoco-warp`` git pin are dropped, so
  resolution no longer breaks when nightly wheels are garbage-collected.

  .. warning::

     ``SimulationCfg.ls_parallel`` is deprecated and now ignored, since parallel
     linesearch was removed upstream in MuJoCo Warp. Setting it emits a
     ``DeprecationWarning``; remove it from any ``SimulationCfg`` you construct.
- Curriculum-mode terrain difficulty is now deterministic across rows
  and reaches the configured ``difficulty_range`` endpoints
  (:issue:`1027`).
- Heightfield terrains now color by absolute height with a diverging palette
  (cool below the ground plane, green at ground level, warm above) on a fixed
  scale, replacing the per-patch normalization. Color is now consistent across
  terrains, and low-amplitude terrain such as ``random_rough`` reads as gently
  tinted ground instead of high-contrast noise.
- ``BoxNestedRingsTerrainCfg`` now builds uniform-height concentric ridges
  whose separating gaps widen with difficulty, replacing the random per-ring
  heights. Rings are colored by height (like the other terrains) and the outer
  border matches the ring height.
- Terrain generation no longer prints timing information to stdout.

Fixed
^^^^^

- Fixed domain randomization events that target different ``axes`` of the same
  model field (e.g. two ``dr.geom_size`` events scaling axis 0 and axis 1
  separately) silently clobbering each other. Each event now writes back only
  the axes it targeted, so per-axis events compose (:issue:`1042`).
- Regenerated the bundled MuJoCo type stubs, which had drifted from the
  installed mujoco version. CI now regenerates them and fails if they are
  stale, so they stay in sync going forward. Run ``make stubs`` to update them
  (:issue:`1048`).
- Fixed ``select_gpus`` crashing when ``CUDA_VISIBLE_DEVICES`` contains MIG
  UUIDs instead of numeric indices.
- Fixed pyramid-stairs terrains (``BoxPyramidStairsTerrainCfg``,
  ``BoxInvertedPyramidStairsTerrainCfg``, and ``BoxOpenStairsTerrainCfg``)
  leaving an empty, geometry-free border at difficulty 0, where the step
  height collapses to zero. The flat border frame is now always generated as
  solid geometry flush with the ground (:issue:`1033`).
- Fixed ``HfPerlinNoiseTerrainCfg`` failing to compile at difficulty 0, where
  the target height collapses to zero and MuJoCo rejects the non-positive
  heightfield size.
- Fixed ``BoxRandomGridTerrainCfg`` producing NaN colors (and failing to build)
  at difficulty 0, where the grid height is zero and the color normalization
  divided by zero.
- Fixed the center platform z-fighting with surrounding geometry in
  ``BoxRandomGridTerrainCfg`` (grid cells were left underneath the platform) and
  ``BoxRandomSpreadTerrainCfg`` (the platform duplicated the floor surface).
- Fixed ``BoxNarrowBeamsTerrainCfg`` square platform corners protruding between
  the beams at high difficulty; the platform now shrinks to stay within the
  beams' angular coverage.
- Fixed ``BoxSteppingStonesTerrainCfg`` reconfiguring abruptly at a difficulty
  threshold, where the stone grid re-tiled as its spacing crossed an integer
  boundary, and leaving an oversized gap around the center platform. The grid is
  now difficulty-independent and the platform snaps to it as a clean island.
- Fixed ``train --video``, ``play``, and ``demo`` crashing with ``OpenGL
  platform library not loaded`` on headless Linux hosts that don't pre-set
  ``MUJOCO_GL``. The default is now applied in ``mjlab/__init__.py`` (Linux
  only) so it takes effect before mujoco's GL backend selection runs.
- Fixed motion tracking re-anchoring to a stale robot pose after a mid-episode
  motion resample. ``MotionCommand._update_command`` now calls ``sim.forward()``
  after resampling so relative body poses read the post-teleport state
  (:issue:`1068`).

Version 1.4.0 (May 26, 2026)
----------------------------

Added
^^^^^

- Added ``BuiltinPdActuator``, the implicit-integration version of
  ``IdealPdActuator``. Same interface (position + velocity targets,
  kp/kd gains), but expresses the PD as native MuJoCo ``<position>``
  and ``<velocity>`` elements so the ``implicit`` / ``implicitfast``
  integrators include the kp/kd derivatives in their velocity update.
  The actuator stays stable at gain/timestep combinations where
  explicit Python PD would diverge, which matters when you want to
  run a real motor's stiff on-board PD gains in sim. ``effort_limit``
  is enforced as a sum-clamp on the two PD terms via
  ``jnt_actfrcrange`` (or ``tendon_actfrcrange``). Supported by
  ``dr.pd_gains`` and ``dr.effort_limits``.
- Added ``mdp.projected_gravity_from_sensor``, an observation that derives
  projected gravity from a ``framezaxis`` up-vector sensor (negated) rather
  than from the root body orientation. Unlike ``mdp.projected_gravity``, it
  reflects the sensor's site frame, so it can observe IMU mounting domain
  randomization (e.g. via ``dr.site_quat``). Go1 and G1 ship an
  ``imu_upvector`` sensor for this.
- Added ``DebugVisualizer.add_box`` for drawing an axis-oriented box
  primitive, mirroring ``add_ellipsoid``. Supported by both the native
  and Viser viewers. ``size`` is the box half-extents (:issue:`992`).
- Added ``--log-root`` CLI option to ``train``, ``play``, and ``evaluate``
  scripts for choosing where training logs are stored. Defaults to
  ``logs/rsl_rl`` (unchanged behavior). Useful for directing outputs to a
  scratch disk or shared mount.
- ``RewardManager``, ``TerminationManager``, and ``MetricsManager`` now
  validate that every term function returns a tensor of shape
  ``(num_envs,)`` when evaluated, raising a clear ``ValueError``
  naming the offending term instead of silently broadcasting or crashing
  with an opaque error later during training.
- Added ``ContactSensor.primary_names`` property to expose the resolved
  primary names in the order they appear along the per-contact axis of the
  output tensors. This makes it possible to map a contact-data column back
  to the primary it belongs to (:issue:`914`).
- Added per-world mesh variant support via ``VariantEntityCfg``. Each
  world in a batched simulation can now use a different mesh asset for
  the same logical entity (e.g. world 0 holds a cube, world 1 a
  sphere). Variants are passed as a ``dict[str, Callable]`` of named
  spec callables; the optional ``assignment`` field controls how worlds
  map to variants and accepts ``None`` (uniform), a ``dict[str, float]``
  of per-variant weights, or a custom ``Callable[[int], Sequence[int]]``.
  Mesh-derived constants (collision bounds, body inertials, subtree
  mass, inverse weights) are compiled per-variant and stored as
  per-world arrays in the Warp model, so domain randomization, the
  native viewer, the offscreen renderer, and the Viser viewer all pick
  up the variant assignment automatically. Variants must share the
  same kinematic structure (same bodies, joints, joint types); only
  mesh geoms may differ. Assignment is fixed at simulation init. See
  :ref:`heterogeneous_worlds` for usage. With help from @XiangruiJiang.
- Per-world mesh variants now support per-variant materials and textures.
  Each variant can reference its own named material, which is automatically
  prefixed and scattered via ``geom_matid`` alongside the existing
  ``geom_dataid`` table. Variants without a material get ``matid = -1``.
  Contribution by @omarrayyann.
- Added ``dr.geom_matid`` to randomize which baked material each geom uses
  per environment, sampling uniformly from ``asset_cfg.material_names``.
  Contribution by @bd-pmorais.

Changed
^^^^^^^

- ``Entity`` now raises a clear error at construction when its spec contains
  more than one freejoint. An entity models a single system rooted at one
  body, so it has at most one freejoint; a second one was previously accepted
  silently and only surfaced later as a cryptic shape mismatch when writing
  root state. Model each detached floating body as its own entry in
  ``SceneCfg.entities`` instead.
- Changed ``compute_root_relative_mpkpe`` to re-anchor the reference to the
  robot's root each step, removing yaw drift as well as translation so it
  measures intrinsic body pose error.
- Changed ``compute_joint_velocity_error`` from an L2 norm to a per-joint
  RMS, so it no longer scales with the number of joints.
- Bumped ``mujoco`` to 3.8 and ``mujoco-warp`` to 3.8.0. The ``multiccd``
  enable flag was removed in mujoco 3.8 (it became default-on), so configs
  that listed ``"multiccd"`` in ``MujocoCfg.enableflags`` need to drop it.
- Camera segmentation now matches ``mujoco_warp``'s typed segmentation
  output. ``CameraSensorData.segmentation`` stores ``(object_id,
  object_type)`` pairs in shape ``[B, H, W, 2]`` instead of the previous
  legacy geom-id-only layout. Contribution by @tkelestemur.
- Sped up ``RayCaster`` post-processing by removing boolean-mask indexing
  operations and replacing them with ``masked_fill_`` plus a clamped-distance
  formulation of ``hit_pos_w`` that places misses at the world origin. This
  removes all CUDA syncs from the ray post-process, letting the CPU thread
  proceed while GPU-based sensing runs. Contribution by @bd-pdomanico.
- Bumped ``rsl-rl-lib`` from 5.0.1 to 5.2.0. This brings ``torch.compile`` support for
  PPO and Distillation, and optional std clamping and constant std in
  ``GaussianDistribution``. No code changes required on the mjlab side.
- ``TerrainEntityCfg`` debug visualization sites (environment origins,
  terrain origins, flat patches) are now off by default. Set
  ``debug_vis=True`` to re-enable them. The sites inflated ``nsite`` and
  caused a measurable slowdown in the per-step ``site_local_to_global``
  kernel (:issue:`942`).
- Task package load failures during ``mjlab`` import now print the full
  traceback (and the entry point's module path) to ``stderr`` instead of
  just the exception message, making it easier to pinpoint the source of
  import errors when running commands like ``list-envs`` (:issue:`910`).
  Contribution by @saikishor.
- Clarified ``ContactSensor`` shape conventions: per-contact fields
  (``found``, ``force``, ``torque``, ``dist``, ``pos``, ``normal``,
  ``tangent``) have shape ``[B, P * num_slots, ...]`` while per-primary
  air-time fields (``current_air_time``, ``last_air_time``,
  ``current_contact_time``, ``last_contact_time``) have shape ``[B, P]``,
  where ``P`` is the number of resolved primaries (:issue:`914`).
- Event functions now share a single ``resolve_env_ids`` helper to expand
  ``env_ids=None`` to all environments, replacing five copies of the same
  guard. ``push_by_setting_velocity`` and ``apply_external_force_torque``
  accept ``env_ids=None`` too, so they work as global-time interval terms.
  Documented when to use ``apply_external_force_torque`` (a constant,
  self-managed wrench) versus ``apply_body_impulse`` (transient, automatic
  impulses) versus ``push_by_setting_velocity`` (an instantaneous velocity
  kick).

Fixed
^^^^^

- Removed use of deprecated ``warp-lang`` symbols (``wp.context.runtime``
  and ``wp.context.Device``) that were dropped in newer ``warp-lang``
  releases, causing ``AttributeError: module 'warp' has no attribute
  'context'`` at import/runtime. mjlab now uses
  ``wp.get_cuda_driver_version()`` and ``wp.Device`` instead
  (:issue:`967`). Contribution by @rdeits.
- Fixed the tracking ``evaluate`` script scoring each metric against the
  next motion frame; the reference is now snapshotted before each step to
  match the reward.
- Fixed the tracking end-effector metrics silently scoring zero for an
  unknown body name; they now raise ``ValueError``.
- Fixed ``compute_mpkpe`` measuring root-relative instead of global error;
  it now uses the global reference ``body_pos_w`` (:issue:`1006`).
- Fixed heavy flicker in offscreen training videos on rough-terrain tasks.
  The renderer recomputed its context "neighbor" robots every frame from
  ``env_origins``, which the terrain curriculum mutates on reset, so the
  neighbor set kept changing and robots popped in and out. The neighbor
  set is now computed once and cached (:issue:`979`).
- Fixed command delay only applying to an actuator's position target.
  ``IdealPdActuator`` and ``DcMotorActuator`` also use velocity and effort, which
  arrived undelayed and out of sync; all command targets now share one delay.
  Zero-reference setups are unaffected.
- Fixed duplicate random seeds across nodes in multi-node training. The
  per-process seed offset in ``scripts/train.py`` now uses the global
  ``RANK`` instead of ``LOCAL_RANK``. Contribution by @bd-pdomanico.
- Fixed ``apply_body_impulse`` firing an impulse on the very first step (and
  the first step after every reset) instead of starting with a cooldown as
  documented. The cooldown is now sampled lazily on the first call so impulse
  timing is decorrelated from episode resets (:issue:`973`).
- Fixed ``dr.pd_gains`` and ``dr.effort_limits`` silently no-oping when
  passed an ``Operation`` object (e.g. ``dr.scale``) instead of a string.
  Both functions now accept ``Operation | str`` like every other DR event
  and raise ``ValueError`` for unsupported operations (:issue:`971`).
- Fixed ``ContactSensor`` with ``global_frame=True`` and
  ``reduce`` ∈ {``"none"``, ``"mindist"``, ``"maxforce"``} producing forces
  rotated onto the wrong axis. The contact-frame→world rotation matrix had
  its columns ordered ``[tangent, tangent2, normal]`` instead of
  ``[normal, tangent, tangent2]``, projecting the normal-force component
  onto a tangent direction. Contribution by @bd-pdomanico.
- Fixed ``extras["log"]`` entries written by reward terms (e.g. ``Metrics/*``
  values in velocity tasks) being silently discarded on any step where at
  least one environment resets. ``_reset_idx`` was clearing the dict after
  ``reward_manager.compute()`` had already populated it. The clear now
  happens at the top of ``step()`` and ``reset()`` so that all entries
  survive (:issue:`957`).
- Fixed ``ContactSensor.compute_first_contact`` and ``compute_first_air``
  occasionally missing events when a contact began or ended right at the
  last physics substep of a control step. ``current_contact_time`` /
  ``current_air_time`` accumulate in float32 and can drift a few ULPs past
  ``dt``, but the default ``abs_tol`` of ``1e-8`` sat at the noise floor
  and rejected the comparison. Raised the default to ``1e-6``, which stays
  well below typical control ``dt`` while comfortably covering float32
  accumulation noise (:issue:`933`). Contribution by @paLeziart.
- Fixed ``out_of_terrain_bounds`` using stale terrain dimensions. It read
  ``TerrainGeneratorCfg.num_cols`` directly, which is ignored in curriculum
  mode (the generator uses ``len(sub_terrains)`` columns instead), and it
  did not account for ``border_width``. The termination now reads the
  effective grid shape from ``terrain.terrain_origins`` and includes the
  border in the footprint, so robots no longer reset while still on valid
  terrain (or fail to reset after running off it) (:issue:`923`).
- ``ObservationManager`` now skips observation groups that end up with
  zero active terms (e.g. all terms set to ``None``) with a log message,
  instead of crashing later in ``torch.stack``/``torch.cat``. This lets
  a shared runner config define groups that become empty under certain
  runtime flags (e.g. model-specific terms all disabled for one variant).
  The whole group can still be set to ``None`` to disable it explicitly.
- Fixed a runtime broadcast error in ``ContactSensor`` when combining
  ``num_slots > 1`` with ``track_air_time=True`` and more than one primary.
  Air-time tracking now reduces ``found`` across slots so that a primary is
  considered in contact when any of its slots reports a match (:issue:`914`).
- Updated the ``create_new_task.ipynb`` Colab tutorial to import
  ``XmlActuatorCfg`` instead of the removed ``XmlVelocityActuatorCfg``.
  Added a regression test (``tests/test_notebooks.py``) that parses each
  notebook cell and verifies that every ``from mjlab... import X``
  reference resolves, so future renames in the mjlab public API can't
  silently rot the tutorials (:issue:`913`).
- Fixed ``ObservationManager`` silently sharing a single ``NoiseModelCfg``
  instance across observation groups that declared terms with the same
  name. ``_group_obs_class_instances`` was keyed by term name alone, so
  the last group processed in ``_prepare_terms`` overwrote earlier
  groups' instances. Symptoms included the wrong noise config being
  applied, shared per-episode state for ``NoiseModelWithAdditiveBias``
  (e.g. bias drawn from the wrong ``bias_noise_cfg``), and missed
  ``reset()`` calls for overwritten instances. Instances are now keyed
  by ``(group_name, term_name)`` so each group owns its own noise model.
- Fixed ``CurriculumManager.get_active_iterable_terms`` raising
  ``TypeError`` when a term's state was a dict. The dict branch indexed
  the output list by term name instead of appending to the local ``data``
  list. No in-tree caller currently invokes this method, so the bug was
  latent.

Version 1.3.0 (April 14, 2026)
------------------------------

Added
^^^^^

- Added ``ManagerBasedRlEnvCfg.auto_reset`` flag. When ``True`` (default),
  ``step()`` continues to reset done environments in place and returns the
  post-reset observation. When ``False``, ``step()`` skips the reset block
  and returns the terminal observation directly; the caller must call
  ``reset(env_ids=...)`` for done environments before the next ``step()``
  or a ``RuntimeError`` is raised. Enables access to the true terminal
  state for algorithms that need it. Note that mjlab's bundled ``train.py``
  uses rsl_rl's ``OnPolicyRunner``, which does not drive manual resets, so
  ``auto_reset=False`` is intended for custom training loops (:issue:`900`).
- Added ``ActuatorCfg.viscous_damping`` for passive velocity proportional
  damping (``f = -b·v``), distinct from the PD derivative gain ``damping``
  used by position and velocity actuators. Maps to ``<joint damping>`` for
  JOINT transmission and ``<tendon damping>`` for TENDON transmission.
  Defaults to ``None`` (preserves the XML value).
- Added :class:`~mjlab.managers.RecorderManager` for logging observations,
  actions, or arbitrary environment data during rollouts. Implement a
  :class:`~mjlab.managers.RecorderTerm` subclass and register it in the
  ``recorders`` dict on ``ManagerBasedRlEnvCfg``. The manager provides
  ``record_pre_reset``, ``record_post_reset``, and ``record_post_step``
  lifecycle hooks with no opinion on how data is stored.
- Added :func:`~mjlab.envs.mdp.curriculums.termination_curriculum` for
  scheduling changes to termination term parameters during training,
  matching the existing ``reward_curriculum`` pattern. Both now share a
  single internal engine with init-time validation of stage ordering,
  field existence, and param keys.
- Added ``reduce`` field to ``MetricsTermCfg``. Setting ``reduce="last"``
  reports the value from the final step of the episode rather than the
  episode mean, which is useful for binary success metrics.
- Added :class:`~mjlab.envs.mdp.actions.RelativeJointPositionAction` for
  joint position control relative to the current configuration. The target is
  ``current_pos + action * scale``, so a zero action holds the current
  configuration rather than commanding the default pose.
- Added :func:`~mjlab.envs.mdp.dr.pair_friction` for randomizing geom-pair
  friction overrides (``pair_friction`` in ``mjModel``), with an
  ``isotropic=True`` option that mirrors the symmetric tangent and roll
  axes so single-axis randomization does not leave the paired axis stale.
- Added ``STAIRS_TERRAINS_CFG`` terrain preset for progressive stair
  curriculum training and ``@terrain_preset`` decorator for composing
  terrain configurations from reusable presets.
- Added cartpole balance and swingup tasks (``Mjlab-Cartpole-Balance`` and
  ``Mjlab-Cartpole-Swingup``) with a :ref:`tutorial <tutorial-cartpole>`
  that walks through building an environment from scratch.
- Added :ref:`motion imitation <motion-imitation>` documentation with
  preprocessing instructions. The README now links here instead of the
  BeyondMimic repository, which produced incompatible NPZ files when used
  with mjlab (:issue:`777`).
- Added ``margin``, ``gap``, and ``solmix`` fields to ``CollisionCfg``
  for per geom contact parameter configuration (:issue:`766`).
- NaN guard now captures mocap body poses (``mocap_pos``, ``mocap_quat``)
  when the model has mocap bodies, enabling full state reconstruction in
  the dump viewer for fixed-base entities.
- Implemented ``ActionTermCfg.clip`` for clamping processed actions after
  scale and offset (:issue:`771`).
- Added ``qfrc_actuator`` and ``qfrc_external`` generalized force accessors
  to ``EntityData``. ``qfrc_actuator`` gives actuator forces in joint space
  (projected through the transmission). ``qfrc_external`` recovers the
  generalized force from body external wrenches (``xfrc_applied``)
  (:issue:`776`).
- Added ``RewardBarPanel`` to the Viser viewer, showing horizontal bars for
  each reward term with a running mean over ~1 second (:issue:`800`).
- Added ``per_substep`` flag to ``MetricsTermCfg`` for evaluating metrics
  once per physics substep inside the decimation loop. The per substep
  values are averaged within each environment step, so episode averages
  remain comparable to regular per step metrics.
- Added ``project-instinct/InstinctMJ`` to the research page's list of
  projects built on mjlab.
- Added a Checkpoints tab to the Viser play viewer for hot-swapping
  checkpoints without restarting. Works with local directories and W&B
  runs (:issue:`751`). Contribution by @omarrayyann.
- Added ``"segmentation"`` camera data type for per-pixel geom ID output
  alongside RGB and depth, and a multi-cube goal-conditioned lifting task
  (``Mjlab-Multi-Cube-Seg-Yam``) that uses it (:issue:`862`).
  Contribution by @pthangeda.

Changed
^^^^^^^

- Renamed the ``list_envs`` console script to ``list-envs`` for consistency
  with the other hyphenated entry points (``viz-nan``, ``export-scene``).
  Invoke via ``uv run list-envs``.
- ``ActuatorCfg.armature`` and ``ActuatorCfg.frictionloss`` now default to
  ``None`` instead of ``0.0``. ``None`` preserves the value defined in the
  XML. Previously, builtin actuators would silently overwrite XML joint and
  tendon properties with zero when these fields were not explicitly set.
  To restore the old behavior, pass ``armature=0.0`` or ``frictionloss=0.0``
  explicitly.
- Actuator delay is now configured inline on any ``ActuatorCfg`` subclass
  (e.g. ``BuiltinPositionActuatorCfg(..., delay_min_lag=2, delay_max_lag=5)``)
  instead of wrapping with ``DelayedActuatorCfg``. ``DelayedActuator``,
  ``DelayedActuatorCfg``, and ``DelayedBuiltinActuatorGroup`` are removed.
- Removed ``delay_target`` from ``ActuatorCfg``. Delay now always applies to
  the actuator's ``command_field`` automatically. Multi-target delay
  (``delay_target=("position", "velocity")``) is no longer supported.
- ``XmlPositionActuatorCfg``, ``XmlVelocityActuatorCfg``, ``XmlMotorActuatorCfg``,
  and ``XmlMuscleActuatorCfg`` are replaced by a single ``XmlActuatorCfg`` that auto
  detects the actuator type from XML. Pass ``command_field=...`` to override detection.
- Replaced the viser viewer internals with the ``mjviser`` package. Scene
  creation, mesh conversion, and overlay rendering (contacts, forces,
  inertia, tendons, joints, frames) are now provided by mjviser. The viewer
  exposes a new Visualization tab for overlay controls and a Groups tab for
  geom/site visibility. Debug visualization and warp tensor conversion remain
  in mjlab's ``MjlabViserScene`` subclass (:issue:`839`).
- In curriculum terrain mode, each terrain type now gets exactly one column
  (``num_cols`` is set to ``len(sub_terrains)``). The ``proportion`` field
  now controls robot spawning distribution across columns rather than column
  count. Random mode is unchanged (:issue:`811`).
- ``BoxSteppingStonesTerrainCfg`` stone size now decreases with difficulty,
  interpolating from the large end of ``stone_size_range`` at difficulty 0
  to the small end at difficulty 1 (:issue:`785`).
- Removed deprecated ``TerrainImporter`` and ``TerrainImporterCfg`` aliases.
  Use ``TerrainEntity`` and ``TerrainEntityCfg`` instead (:issue:`667`).
- ``Entity.clear_state()`` is deprecated. Use ``Entity.reset()`` instead.
  ``clear_state`` only zeroed actuator targets without resetting actuator
  internal state (e.g. delay buffers), which could cause stale commands
  after teleporting the robot to a new pose.
- Removed ``EntityData.generalized_force``. The property was bugged (indexed
  free joint DOFs instead of articulated DOFs) and the name was ambiguous.
  Use ``qfrc_actuator`` or ``qfrc_external`` instead (:issue:`776`).
- ``get_wandb_checkpoint_path`` now filters checkpoints server-side via the
  ``pattern`` parameter, avoiding unnecessary pagination and tolerance to
  corrupted metadata (:issue:`898`).

Fixed
^^^^^

- ``train`` and ``play`` now print a top-level usage message when invoked
  with ``-h`` / ``--help`` and no task argument, pointing users at
  ``list-envs`` and ``<TASK> --help`` (:issue:`905`).
- Fixed ghost geom filtering in the Viser viewer. Ghost geoms were selected
  by collision flags, so collision-disabled robot geoms appeared as ghosts.
  The viewer now uses visual alpha to determine which geoms to render.
- Scene now warns when an attached entity or terrain spec has non-default
  ``<option>`` fields (e.g. ``<flag contact="disable"/>``), which are
  silently dropped by ``MjSpec.attach()``. Use ``MujocoCfg`` to set
  simulation options instead (:issue:`885`).
- Fixed ``SceneEntityCfg`` names and IDs ordering mismatch when
  ``preserve_order=False`` (:issue:`876`). Contribution by @jsw7460.
- Fixed ONNX export path resolution in the velocity, manipulation, and
  tracking runners when a parent directory name contains the word
  ``"model"`` (:issue:`867`). Contribution by @gokulp01.
- ``export-scene`` now writes only referenced assets and places them
  correctly under the output directory. Previously, asset keys containing
  path traversal could write files outside the output directory, and all
  spec assets were included regardless of whether the scene XML referenced
  them (:issue:`858`).
- ``electrical_power_cost`` now uses ``qfrc_actuator`` (joint space) instead
  of ``actuator_force`` (actuation space) for mechanical power computation.
  Previously the reward was incorrect for actuators with gear ratios other
  than 1 (:issue:`776`).
- ``create_velocity_actuator`` no longer sets ``ctrllimited=True`` with
  ``inheritrange=1.0``. This caused a ``ValueError`` for continuous joints
  (e.g. wheels) that have no position range defined (:issue:`787`).
- ``write_root_com_velocity_to_sim`` no longer fails with tensor ``env_ids``
  on floating base entities (:issue:`793`).
- Joint limits for unlimited joints are now set to [-inf, inf] instead of
  [0, 0]. Previously the zero range caused incorrect clamping for entities
  with unlimited hinge or slide joints.
- Contact force visualization now copies ``ctrl`` into the CPU ``MjData``
  before calling ``mj_forward``. Actuators that compute torques in Python
  (``DcMotorActuator``, ``IdealPdActuator``) previously showed incorrect
  contact forces because the viewer ran with ``ctrl=0``
  (:issue:`786`).
- ``BoxSteppingStonesTerrainCfg`` no longer creates a large gap around the
  platform. Stones are now only skipped when their center falls inside the
  platform; edges that extend under the platform are allowed since the
  platform covers them (:issue:`785`).
- ``dr.pseudo_inertia`` no longer loads cuSOLVER, eliminating ~4 GB of
  persistent GPU memory overhead. Cholesky and eigendecomposition are now
  computed analytically for the small matrices involved (4x4 and 3x3)
  (:issue:`753`).
- Set terrain geom mass to zero so that the static terrain body does not
  inflate ``stat.meanmass``, which made force arrow visualization invisible
  on rough terrain (:issue:`734`, :issue:`537`).
- Native viewer now syncs ``qpos0`` when domain randomized, fixing incorrect
  body positions after ``dr.joint_default_pos`` randomization
  (:issue:`760`).
- ``command_manager.compute()`` is now called during ``reset()`` so that
  derived command state (e.g. relative body positions in tracking
  environments) is populated before the first observation is returned
  (:issue:`761`).
- ``RayCastSensor`` with ``ray_alignment="yaw"`` or ``"world"`` now correctly
  aligns the frame offset when attached to a site or geom with a local offset
  from its parent body. Previously only ray directions and pattern offsets were
  aligned, causing the frame position to swing with body pitch/roll
  (:issue:`775`).

Version 1.2.0 (March 6, 2026)
-----------------------------

.. admonition:: Breaking API changes
   :class: attention

   - ``randomize_field`` no longer exists. Replace calls with typed functions
     from the new ``dr`` module (e.g. ``dr.geom_friction``, ``dr.body_mass``).
   - ``EventTermCfg`` no longer accepts ``domain_randomization``. The
     ``@requires_model_fields`` decorator on each ``dr`` function takes care
     of field expansion automatically.
   - ``Scene.to_zip()`` is deprecated. Use ``Scene.write(path, zip=True)``.
   - ``RslRlModelCfg`` no longer accepts ``stochastic``, ``init_noise_std``,
     or ``noise_std_type``. Use ``distribution_cfg`` instead
     (e.g. ``{"class_name": "GaussianDistribution", "init_std": 1.0,
     "std_type": "scalar"}``). Existing checkpoints are automatically
     migrated on load.

Added
^^^^^

- Added ``"step"`` event mode that fires every environment step.
- Added ``apply_body_impulse`` event for applying transient external wrenches
  to bodies with configurable duration and optional application point offset.
- ONNX auto-export and metadata attachment for manipulation tasks (lift cube)
  on every checkpoint save, matching the velocity and tracking task behavior.
- Multi-frame ``RayCastSensor``: pass a tuple of ``ObjRef`` to ``frame`` for
  per-site raycasting with independent body exclusion. New properties:
  ``num_frames``, ``num_rays_per_frame``. New ``RayCastData`` fields:
  ``frame_pos_w`` and ``frame_quat_w``.
- ``RingPatternCfg`` ray pattern for concentric ring sampling around each
  frame.
- ``TerrainHeightSensor``, a ``RayCastSensor`` subclass that computes
  per-frame vertical clearance above terrain (``sensor.data.heights``).
  Velocity task configs now use it for ``feet_clearance``,
  ``feet_swing_height``, and ``foot_height``, replacing the previous
  world-Z proxy that was incorrect on rough terrain.
- Cloud training support via `SkyPilot <https://skypilot.readthedocs.io/>`_
  and Lambda Cloud, with documentation covering setup, monitoring, and
  cost management.
- W&B hyperparameter sweep scripts that distribute one agent per GPU
  across a multi-GPU instance.
- Contributing guide with documentation for shared Claude Code commands
  (``/update-mjwarp``, ``/commit-push-pr``).
- Added optional ``ViewerConfig.fovy`` and apply it in native viewer camera
  setup when provided.
- Native viewer now tracks the first non-fixed body by default (matching
  the Viser viewer behavior introduced in
  ``716aaaa58ad7bfaf34d2f771549d461204d1b4ba``).
- New ``dr`` module (``mjlab.envs.mdp.dr``) replacing ``randomize_field``
  with typed per-field domain randomization functions. Each function
  automatically recomputes derived fields via ``set_const``. Highlights:

  - Camera and light randomization: ``dr.cam_fovy``, ``dr.cam_pos``,
    ``dr.cam_quat``, ``dr.cam_intrinsic``, ``dr.light_pos``,
    ``dr.light_dir``. Camera and light names are now supported in
    ``SceneEntityCfg`` (``camera_names`` / ``light_names``).
  - ``dr.pseudo_inertia`` for physics-consistent randomization of
    ``body_mass``, ``body_ipos``, ``body_inertia``, and ``body_iquat``
    via the pseudo-inertia matrix parameterization (Rucker & Wensing
    2022). Replaces the removed ``dr.body_inertia`` /
    ``dr.body_iquat``.
  - ``dr.geom_size`` with automatic recomputation of ``geom_rbound``
    and ``geom_aabb`` for broadphase consistency.
  - ``dr.tendon_armature`` and ``dr.tendon_frictionloss``.
  - ``dr.body_quat``, ``dr.geom_quat``, and ``dr.site_quat`` with RPY
    perturbation composed onto the default quaternion.
  - Extensible ``Operation`` and ``Distribution`` types. Users can define
    custom operations and distributions as class instances and pass them
    anywhere a string is accepted. Built-in instances (``dr.abs``,
    ``dr.scale``, ``dr.add``, ``dr.uniform``, ``dr.log_uniform``,
    ``dr.gaussian``) are exported from the ``dr`` module.
  - ``dr.mat_rgba`` for per-world material color randomization. Tints
    the texture color, useful for randomizing appearance of textured
    surfaces. Material names are now supported in ``SceneEntityCfg``
    (``material_names``).
  - Fixed ``dr.effort_limits`` drifting on repeated randomization.
  - Fixed ``dr.body_com_offset`` not triggering ``set_const``.

- ``export-scene`` CLI script to export any task scene or asset_zoo entity
  (``g1``, ``go1``, ``yam``) to a directory or zip archive for inspection
  and debugging.

- ``yam_lift_cube_vision_env_cfg`` now randomizes cube color (``dr.geom_rgba``)
  on every reset when ``cam_type="rgb"``.

- The native viewer now reflects per-world DR changes to visual model fields
  on each reset. Geom appearance, body and site poses, camera parameters,
  and light positions are all synced from the GPU model before rendering.
  Inertia boxes (press ``I``) and camera frustums (press ``Q``) update
  correctly when the corresponding fields are randomized. See
  :doc:`randomization` for viewer-specific caveats.

- ``MaterialCfg.geom_names_expr`` for assigning materials to geoms by
  name pattern during ``edit_spec``.

- ``TerrainEntityCfg`` now exposes ``textures``, ``materials``, and
  ``lights`` as configurable fields (previously hardcoded). Set
  ``textures=()``, ``materials=()`` to use flat ``dr.geom_rgba``
  instead of the default checker texture.

- ``DebugVisualizer`` now supports ellipsoid visualization via
  ``add_ellipsoid``.

- Interactive velocity joystick sliders in the Viser viewer. Enable the
  joystick under Commands/Twist to override velocity commands with manual
  sliders for ``lin_vel_x``, ``lin_vel_y``, and ``ang_vel_z``
  (`#666 <https://github.com/mujocolab/mjlab/issues/666>`_).
- Per-term debug visualization toggles in the Viser viewer. Individual
  command term visualizers (e.g. velocity arrows) can now be toggled
  independently under Scene/Debug Viz.
- Viewer single-step mode: press RIGHT arrow (native) or click "Step"
  (Viser) to advance exactly one physics step while paused.
- Viewer error recovery: exceptions during stepping now pause the viewer
  and log the traceback instead of crashing the process.
- Native viewer runs forward kinematics while paused, keeping
  perturbation visuals accurate.
- Viewer speed multipliers use clean power-of-2 fractions (1/32x to 1x).

- Visualizers display the realtime factor alongside FPS.

- ``joint_torques_l2`` now respects ``SceneEntityCfg.actuator_ids``,
  allowing penalization of a subset of actuators instead of all of them
  (`#703 <https://github.com/mujocolab/mjlab/pull/703>`_). Contribution by
  `@saikishor <https://github.com/saikishor>`_.

- Terrain is now a proper ``Entity`` subclass (``TerrainEntity``). This
  allows domain randomization functions to target terrain parameters
  (friction, cameras, lights) via ``SceneEntityCfg("terrain", ...)``.
  ``TerrainImporter`` / ``TerrainImporterCfg`` remain as aliases but will be
  deprecated in a future version.
- Added ``upload_model`` option to ``RslRlBaseRunnerCfg`` to control W&B model
  file uploads (``.pt`` and ``.onnx``) while keeping metric logging enabled
  (`#654 <https://github.com/mujocolab/mjlab/pull/654>`_).
- ``Scene.write(output_dir, zip=False)`` exports the scene XML and mesh
  assets to a directory (or zip archive). Replaces ``Scene.to_zip()``.
- ``Entity.write_xml()`` and ``Scene.write()`` now apply XML fixups
  (empty defaults, duplicate nested defaults) and strip buffer textures
  that ``MjSpec.to_xml()`` cannot serialize.
- ``fix_spec_xml`` and ``strip_buffer_textures`` utilities in
  ``mjlab.utils.xml``.

Changed
^^^^^^^

- Native viewer now syncs ``xfrc_applied`` to the render buffer and draws
  arrows for any nonzero applied forces. Mouse perturbation forces are
  converted to ``qfrc_applied`` (generalized joint space) so they coexist
  with programmatic forces on ``xfrc_applied`` without conflict.
- ``ViewerConfig.OriginType.WORLD`` now configures a free camera at the
  specified lookat point instead of auto tracking a body. A new ``AUTO``
  origin type (now the default) preserves the previous auto tracking
  behavior.
- Upgraded ``rsl-rl-lib`` from 4.0.1 to 5.0.1. ``RslRlModelCfg`` now
  uses ``distribution_cfg`` dict instead of ``stochastic`` /
  ``init_noise_std`` / ``noise_std_type``. Existing checkpoints are
  automatically migrated on load.
- Reorganized the Viser Controls tab into a cleaner folder hierarchy:
  Info, Simulation, Commands, Scene (with Environment, Camera, Debug Viz,
  Contacts sub-folders), and Camera Feeds. The Environment folder is
  hidden for single-env tasks and the Commands folder is hidden when no
  command terms are active.
- Viser camera tracking is now enabled by default so the agent stays in
  frame on launch.
- Self collision and illegal contact sensors now use ``history_length`` to
  catch contacts across decimation substeps. Reward and termination functions
  read ``force_history`` with a configurable ``force_threshold``.
- Replaced the single ``scale`` parameter in ``DifferentialIKActionCfg`` with
  separate ``delta_pos_scale`` and ``delta_ori_scale`` for independent scaling
  of position and orientation components.
- Improved offscreen multi environment framing by selecting neighboring
  environments around the focused env instead of first N envs.
- Tuned tracking task viewer defaults for tighter camera framing.
- Disabled shadow casting on the G1 tracking light to avoid duplicate
  stacked shadows when robots are close.

Fixed
^^^^^

- Fixed actuator target resolution for entities whose ``spec_fn`` uses
  internal ``MjSpec.attach(prefix=...)``
  (`#709 <https://github.com/mujocolab/mjlab/issues/709>`_).
- Fixed viewer physics loop starving the renderer by replacing the single
  sim-time budget with a two-clock design (tracked vs actual sim time).
  Physics now self-corrects after overshooting, keeping FPS smooth at all
  speed multipliers.
- Bundled ``ffmpeg`` for ``mediapy`` via ``imageio-ffmpeg``, removing the
  requirement for a system ``ffmpeg`` install. Thanks to
  `@rdeits-bd <https://github.com/rdeits-bd>`_ for the suggestion.
- Fixed ``height_scan`` returning ~0 for missed rays; now defaults to
  ``max_distance``. Replaced ``clip=(-1, 1)`` with ``scale`` normalization
  in the velocity task config. Thanks to `@eufrizz <https://github.com/eufrizz>`_
  for reporting and the initial fix (`#642 <https://github.com/mujocolab/mjlab/pull/642>`_).
- Fixed ghost mesh visualization for fixed-base entities by extending
  ``DebugVisualizer.add_ghost_mesh`` to optionally accept ``mocap_pos`` and
  ``mocap_quat`` (`#645 <https://github.com/mujocolab/mjlab/pull/645>`_).
- Fixed viser viewer crashing on scenes with no mocap bodies by adding
  an ``nmocap`` guard, matching the native viewer behavior.
- Fixed offscreen rendering artifacts in large vectorized scenes by applying
  a render local extent override in ``OffscreenRenderer`` and restoring the
  original extent on close.
- Fixed ``RslRlVecEnvWrapper.unwrapped`` to return the base environment,
  ensuring checkpoint state restore and logging work correctly when wrappers
  such as ``VideoRecorder`` are enabled.

Version 1.1.1 (February 14, 2026)
---------------------------------

Added
^^^^^

- Added reward term visualization to the native viewer (toggle with ``P``) (`#629 <https://github.com/mujocolab/mjlab/pull/629>`_).
- Added ``DifferentialIKAction`` for task-space control via damped
  least-squares IK. Supports weighted position/orientation tracking,
  soft joint-limit avoidance, and null-space posture regularization.
  Includes an interactive viser demo (``scripts/demos/differential_ik.py``) (`#632 <https://github.com/mujocolab/mjlab/pull/632>`_).

Fixed
^^^^^

- Fixed ``play.py`` defaulting to the base rsl-rl ``OnPolicyRunner`` instead
  of ``MjlabOnPolicyRunner``, which caused a ``TypeError`` from an unexpected
  ``cnn_cfg`` keyword argument (`#626 <https://github.com/mujocolab/mjlab/pull/626>`_). Contribution by
  `@griffinaddison <https://github.com/griffinaddison>`_.

Changed
^^^^^^^

- Removed ``body_mass``, ``body_inertia``, ``body_pos``, and ``body_quat``
  from ``FIELD_SPECS`` in domain randomization. These fields have derived
  quantities that require ``set_const`` to recompute; without that call,
  randomizing them silently breaks physics (`#631 <https://github.com/mujocolab/mjlab/pull/631>`_).
- Replaced ``moviepy`` with ``mediapy`` for video recording. ``mediapy``
  handles cloud storage paths (GCS, S3) natively (`#637 <https://github.com/mujocolab/mjlab/pull/637>`_).

.. figure:: _static/changelog/native_reward.png
   :width: 80%

Version 1.1.0 (February 12, 2026)
---------------------------------

Added
^^^^^

- Added RGB and depth camera sensors and BVH-accelerated raycasting (`#597 <https://github.com/mujocolab/mjlab/pull/597>`_).
- Added ``MetricsManager`` for logging custom metrics during training (`#596 <https://github.com/mujocolab/mjlab/pull/596>`_).
- Added terrain visualizer (`#609 <https://github.com/mujocolab/mjlab/pull/609>`_). Contribution by
  `@mktk1117 <https://github.com/mktk1117>`_.

.. figure:: _static/changelog/terrain_visualizer.jpg
   :width: 80%

- Added many new terrains including ``HfDiscreteObstaclesTerrainCfg``,
  ``HfPerlinNoiseTerrainCfg``, ``BoxSteppingStonesTerrainCfg``,
  ``BoxNarrowBeamsTerrainCfg``, ``BoxRandomStairsTerrainCfg``, and
  more. Added flat patch sampling for heightfield terrains (`#542 <https://github.com/mujocolab/mjlab/pull/542>`_, `#581 <https://github.com/mujocolab/mjlab/pull/581>`_).
- Added site group visualization to the Viser viewer (Geoms and Sites
  tabs unified into a single Groups tab) (`#551 <https://github.com/mujocolab/mjlab/pull/551>`_).
- Added ``env_ids`` parameter to ``Entity.write_ctrl_to_sim`` (`#567 <https://github.com/mujocolab/mjlab/pull/567>`_).

Changed
^^^^^^^

- Upgraded ``rsl-rl-lib`` to 4.0.0 and replaced the custom ONNX
  exporter with rsl-rl's built-in ``as_onnx()`` (`#589 <https://github.com/mujocolab/mjlab/pull/589>`_, `#595 <https://github.com/mujocolab/mjlab/pull/595>`_).
- ``sim.forward()`` is now called unconditionally after the decimation
  loop. See :ref:`faq-sim-forward` for details (`#591 <https://github.com/mujocolab/mjlab/pull/591>`_).
- Unnamed freejoints are now automatically named to prevent
  ``KeyError`` during entity init (`#545 <https://github.com/mujocolab/mjlab/pull/545>`_).

Fixed
^^^^^

- Fixed ``randomize_pd_gains`` crash with ``num_envs > 1`` (`#564 <https://github.com/mujocolab/mjlab/pull/564>`_).
- Fixed ``ctrl_ids`` index error with multiple actuated entities (`#573 <https://github.com/mujocolab/mjlab/pull/573>`_).
  Reported by `@bwrooney82 <https://github.com/bwrooney82>`_.
- Fixed Viser viewer rendering textured robots as gray (`#544 <https://github.com/mujocolab/mjlab/pull/544>`_).
- Fixed Viser plane rendering ignoring MuJoCo size parameter (`#540 <https://github.com/mujocolab/mjlab/pull/540>`_).
- Fixed ``HfDiscreteObstaclesTerrainCfg`` spawn height (`#552 <https://github.com/mujocolab/mjlab/pull/552>`_).
- Fixed ``RaycastSensor`` visualization ignoring the all-envs toggle (`#607 <https://github.com/mujocolab/mjlab/pull/607>`_).
  Contribution by `@oxkitsune <https://github.com/oxkitsune>`_.

Version 1.0.0 (January 28, 2026)
--------------------------------

Initial release of mjlab.
