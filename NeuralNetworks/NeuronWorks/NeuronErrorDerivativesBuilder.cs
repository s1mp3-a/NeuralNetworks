using System;

namespace NeuralNetworks.NeuronWorks
{
    internal class NeuronErrorDerivativesBuilder
    {
        public Func<double, double> Sigmoid { get; } =
            x => x * (1 - x);
        public Func<double, double> Linear { get; } =
            x => 1;
        public Func<double, double> Tanh { get; } =
            x => 1 - x * x;
    }
}