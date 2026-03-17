// The TrainParams schema the user builds in the UI
export interface TrainParams {
    loss: { type: string }; // The string comes from the API
    metrics: { type: string }[] | null; // The strings come from the API
    epochs: number;
    batchsize: number;
}

export interface GraphPayload {
    nodes: { id: string; kind: Record<string, unknown>; position: { x: number; y: number } }[];
    edges: { id: string; source: string; target: string }[];
}

// The exact wrapper payload the Rust backend expects for the POST request
export interface TrainRequestPayload {
    graph: GraphPayload;
    params: TrainParams;
}
