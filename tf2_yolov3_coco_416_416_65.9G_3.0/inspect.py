import xir

graph = xir.Graph.deserialize("compiled_model/yolov3_tf2.xmodel")
subgraphs = [s for s in graph.get_root_subgraph().children if s.has_attr("device")]

for sg in subgraphs:
    print(f"\n=== Subgraph: {sg.get_name()} ===")

    print("Inputs:")
    for t in sg.get_inputs():
        print(f"  - {t.get_name()}, shape={t.get_shape()}")

    print("Outputs:")
    for t in sg.get_outputs():
        print(f"  - {t.get_name()}, shape={t.get_shape()}")

