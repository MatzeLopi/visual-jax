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

export interface Model {
    model_id: string;
    user_id: string;
    version_: number;
    model_name: string | null;
    model_description: string | null;
    model_path: string;
}

export interface Log {
    origin: string;
    text: string;
    created_at: string | null;
}
