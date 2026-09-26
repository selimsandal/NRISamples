// © 2026 NVIDIA Corporation

struct Input {
    float2 position : POSITION;
    float2 instanceOffset : TEXCOORD0;
    float4 color : COLOR0;
};

struct Output {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

Output main(Input input) {
    Output output;
    output.position = float4(input.position + input.instanceOffset, 0.0, 1.0);
    output.color = input.color;
    return output;
}
