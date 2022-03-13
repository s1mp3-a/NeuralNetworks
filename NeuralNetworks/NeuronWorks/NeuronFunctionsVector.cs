using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks.NeuronWorks;

internal class NeuronFunctionsVector
{
    public Func<Vector<double>, Vector<double>> Sigmoid { get; } =
        (v) => 1 / (1 + (-v).PointwiseExp());
}