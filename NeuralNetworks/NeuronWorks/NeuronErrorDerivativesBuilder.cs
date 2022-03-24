using System;

namespace NeuralNetworks.NeuronWorks
{
    internal class NeuronErrorDerivativesBuilder
    {
        public Func<double, double> Sigmoid { get; } =
            (x) => x * (1 - x);

        public Func<double, double> Linear { get; } =
            x => -2d * x;
    }
}