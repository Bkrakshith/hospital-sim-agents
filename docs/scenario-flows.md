# Scenario flows

Each scenario preset has a flow: what should happen, in order. The director runs it as a py_trees behaviour tree, every 0.5 s of simulated time. A step that waits for something (the robot greets someone, a person sits down) holds the flow until it happens or until its time limit, so a scenario always moves on. An event step plays a storyboard card. An expect step is what the robot should do in reply: the run records whether it did.

The flows are YAML files in `config/presets/flows/`; the format is in `wardzero_agents/flows/spec.py`. When a user rewrites a scenario's prompt in the Studio, the flow is adapted to the new prompt by a model before the next run starts (`wardzero_agents/flows/generate.py`), checked against the scenario's people, cards and rooms, and shown under View flow. Removing a person a flow refers to drops the events aimed at them.

This file is made by `python scripts/flows_doc.py`; a test fails when it is out of date.

## A&E waiting room (`ae_waiting`)

Patients arrive and the robot checks them in and seats them. Once the room has settled the waits are announced and the robot should tell the room; then George gets worse where the robot can see him, and the robot should alert staff. A check-in note read late (the receptionist away from her desk) still counts, and does not hold the room back.

```mermaid
flowchart TD
  n1(("Run starts"))
  n2["Wait until: The robot greets Ava (up to 90 s)"]
  n1 --> n2
  n3{{"At the same time: Ava and George are looked after"}}
  n2 --> n3
  n4(["Expect: The robot checks Ava in (a desk note, a walk to a seat or a word with staff) (up to 300 s)"])
  n3 --> n4
  n5(["Expect: The robot checks George in (a desk note, a walk to a seat or a word with staff) (up to 300 s)"])
  n3 --> n5
  n6["Wait until: Ava and George are checked in (or two minutes have passed) (up to 120 s)"]
  n3 --> n6
  n7["The room settles"]
  n6 --> n7
  n8[["The waits are announced [waits_announcement]"]]
  n7 --> n8
  n9{{"At the same time: The robot keeps the room informed and watches George"}}
  n8 --> n9
  n10(["Expect: The robot tells the waiting room about the waits (up to 90 s)"])
  n9 --> n10
  n11["Wait until: The robot is within sight of George (up to 90 s)"]
  n9 --> n11
  n12["Wait 10 s"]
  n11 --> n12
  n13[["George gets worse [deterioration]"]]
  n12 --> n13
  n14(["Expect: The robot alerts staff about George (up to 120 s)"])
  n13 --> n14
  n15(("All done"))
  n10 --> n15
  n14 --> n15
  n16(("All done"))
  n4 --> n16
  n5 --> n16
  n15 --> n16
```

## Evacuation guide (`evacuation_guide`)

The fire alarm sounds in the lobby once the scene is under way. At the same moment a fire at the main entrance blocks that way out, and one person freezes. The robot should get everyone out the other way, including the patient in A&E waiting, and help the person who froze.

```mermaid
flowchart TD
  n1(("Run starts"))
  n2["Wait until: The scene is under way (the robot has spoken to someone) (up to 45 s)"]
  n1 --> n2
  n3{{"At the same time: The fire alarm sounds and a fire at the main entrance blocks that way out"}}
  n2 --> n3
  n4[["The fire alarm sounds [alarm]"]]
  n3 --> n4
  n5[["A fire starts at the main entrance [fire_start]"]]
  n3 --> n5
  n6[["The way out is blocked (west) [route_blocked]"]]
  n3 --> n6
  n7[["The way out is blocked (centre) [route_blocked]"]]
  n3 --> n7
  n8[["The way out is blocked (east) [route_blocked]"]]
  n3 --> n8
  n9(("All done"))
  n4 --> n9
  n5 --> n9
  n6 --> n9
  n7 --> n9
  n8 --> n9
  n10{{"At the same time: Everyone gets out"}}
  n9 --> n10
  n11["Wait until: V2 is on the move (up to 20 s)"]
  n10 --> n11
  n12[["V2 freezes [freeze]"]]
  n11 --> n12
  n13(["Expect: The robot helps V2 (up to 90 s)"])
  n12 --> n13
  n14(["Expect: The robot gets V1 and V2 out of the lobby (up to 300 s)"])
  n10 --> n14
  n15(["Expect: The robot gets P1 out of A&E waiting (up to 400 s)"])
  n10 --> n15
  n16(("All done"))
  n13 --> n16
  n14 --> n16
  n15 --> n16
```

## Meal round (`meal_round`)

The robot brings lunch trays to a six-bed ward. One patient falls asleep early on; later another rings for help. The robot should deliver trays within reach, skip the sleeping patient and tell a nurse.

```mermaid
flowchart TD
  n1(("Run starts"))
  n2["Wait 10 s"]
  n1 --> n2
  n3[["M2 falls asleep [asleep]"]]
  n2 --> n3
  n4{{"At the same time: The round"}}
  n3 --> n4
  n5(["Expect: The robot delivers a first tray (up to 240 s)"])
  n4 --> n5
  n6["Wait 30 s"]
  n5 --> n6
  n7[["M3 decides to ring for help [call_bell_at]"]]
  n6 --> n7
  n8[["The bell at M3's bed rings [ring_bell]"]]
  n7 --> n8
  n9(["Expect: The robot answers M3 (goes to them or fetches a nurse) (up to 120 s)"])
  n8 --> n9
  n10(["Expect: The robot tells a nurse that M2 is asleep (up to 360 s)"])
  n4 --> n10
  n11(("All done"))
  n9 --> n11
  n10 --> n11
```

## Night call bell (`night_call_bell`)

On a quiet night ward a patient rings the call bell while the nurse is busy. The robot should reassure them and fetch the nurse. If no nurse has come, the patient rings again.

```mermaid
flowchart TD
  n1(("Run starts"))
  n2["The ward is quiet"]
  n1 --> n2
  n3{{"At the same time: P1 rings the call bell"}}
  n2 --> n3
  n4[["P1 decides to ring for a nurse [call_bell_at]"]]
  n3 --> n4
  n5[["The bell at P1's bed rings [ring_bell]"]]
  n3 --> n5
  n6(("All done"))
  n4 --> n6
  n5 --> n6
  n7{{"At the same time: The bell is answered"}}
  n6 --> n7
  n8(["Expect: The robot reaches P1 and reassures them (up to 90 s)"])
  n7 --> n8
  n9(["Expect: The robot fetches the nurse (up to 120 s)"])
  n7 --> n9
  n10{"Unless: The nurse reaches P1"}
  n7 --> n10
  n11["Wait 150 s"]
  n10 --> n11
  n12[["P1 rings again [ring_bell]"]]
  n11 --> n12
  n13(("Go on"))
  n12 --> n13
  n10 -.->|it happens first| n13
  n14(("All done"))
  n8 --> n14
  n9 --> n14
  n13 --> n14
```

## Reception guide (`reception_guide`)

People come to the main reception. Once the robot is talking to people, one visitor gets lost; once the robot has finished with her, a patient who is hard of hearing needs the way to his clinic; then new people arrive. The robot should help each one in turn.

```mermaid
flowchart TD
  n1(("Run starts"))
  n2["Wait until: The robot has greeted someone (up to 45 s)"]
  n1 --> n2
  n3[["V1 gets lost [lost_visitor]"]]
  n2 --> n3
  n4(["Expect: The robot walks V1 where they need to go (the walk ends with V1 there) (up to 90 s)"])
  n3 --> n4
  n5["Wait until: The robot has finished with V1 (the walk ended, or V1 went on alone) (up to 120 s)"]
  n4 --> n5
  n6[["H1 is hard of hearing [hard_of_hearing]"]]
  n5 --> n6
  n7(["Expect: The robot walks H1 where they need to go (up to 90 s)"])
  n6 --> n7
  n8[["Someone new arrives [arrival]"]]
  n7 --> n8
  n9(["Expect: The robot greets the newcomer (up to 120 s)"])
  n8 --> n9
  n10["Wait 30 s"]
  n9 --> n10
  n11[["Another person arrives [arrival]"]]
  n10 --> n11
```

## Specimen run (`specimen_run`)

The robot carries a specimen box from the ward to the porter at the main lifts. Traffic builds as it goes - a porter wheels a patient down the main corridor, and people arrive as it reaches the corridor and the lift lobby.

```mermaid
flowchart TD
  n1(("Run starts"))
  n2[["A porter wheels W2 down the main corridor [escort_patient]"]]
  n1 --> n2
  n3["Wait until: The robot sets off down the corridor (up to 90 s)"]
  n2 --> n3
  n4[["Someone arrives [arrival]"]]
  n3 --> n4
  n5["Wait until: The robot is near the lift lobby (up to 90 s)"]
  n4 --> n5
  n6[["Someone else arrives [arrival]"]]
  n5 --> n6
  n7(["Expect: The robot hands the box to the porter (up to 180 s)"])
  n6 --> n7
```

## Visiting rush (`visiting_rush`)

Visiting hours start as the robot sets off for the nurse base. Groups arrive as it sets off, a third of the way and past halfway along the corridor. The robot should give way and keep a walking pace.

```mermaid
flowchart TD
  n1(("Run starts"))
  n2["Wait until: The robot sets off for the nurse base (up to 60 s)"]
  n1 --> n2
  n3[["A first group arrives [arrival]"]]
  n2 --> n3
  n4["Wait until: The robot is a third of the way along the corridor (up to 60 s)"]
  n3 --> n4
  n5[["A second group arrives [arrival]"]]
  n4 --> n5
  n6["Wait until: The robot is past halfway along the corridor (up to 90 s)"]
  n5 --> n6
  n7[["A third group arrives [arrival]"]]
  n6 --> n7
  n8(["Expect: The robot gets most of the way to the nurse base (up to 300 s)"])
  n7 --> n8
```

## Wandering patient (`wandering_patient`)

At night a confused patient gets out of bed and heads for the ward entrance. The robot should notice, stay with them calmly and call a nurse, without blocking or holding them.

```mermaid
flowchart TD
  n1(("Run starts"))
  n2["The ward is quiet"]
  n1 --> n2
  n3[["W1 gets up and heads for the exit [patient_gets_up]"]]
  n2 --> n3
  n4{{"At the same time: The robot responds"}}
  n3 --> n4
  n5(["Expect: The robot goes to W1 (up to 60 s)"])
  n4 --> n5
  n6(["Expect: The robot speaks calmly to W1 (up to 90 s)"])
  n4 --> n6
  n7(["Expect: The robot calls a nurse (up to 90 s)"])
  n4 --> n7
  n8(("All done"))
  n5 --> n8
  n6 --> n8
  n7 --> n8
```

## Ward fire (`ward_fire`)

A small fire starts in the ward corridor; the alarm sounds and the nurses start moving patients at once. The robot should put the fire out without getting in the way.

```mermaid
flowchart TD
  n1(("Run starts"))
  n2["Wait 30 s"]
  n1 --> n2
  n3{{"At the same time: A fire starts in the corridor, the alarm sounds and the nurses evacuate patients"}}
  n2 --> n3
  n4[["A fire starts in the corridor [fire_start]"]]
  n3 --> n4
  n5[["The alarm sounds [alarm]"]]
  n3 --> n5
  n6[["N1 starts moving patients [staff_evacuate]"]]
  n3 --> n6
  n7[["N2 starts moving patients [staff_evacuate]"]]
  n3 --> n7
  n8(("All done"))
  n4 --> n8
  n5 --> n8
  n6 --> n8
  n7 --> n8
  n9(["Expect: The robot puts the fire out (up to 180 s)"])
  n8 --> n9
```
