// © 2026 NVIDIA Corporation

float4 main(nointerpolation uint instance : TEXCOORD0) : SV_Target {
    return instance == 0 ? float4(1, 0, 0, 1) : float4(0, 1, 0, 1);
}
