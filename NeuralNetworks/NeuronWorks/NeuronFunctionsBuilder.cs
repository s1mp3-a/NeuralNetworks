using System;

namespace NeuralNetworks.NeuronWorks
{
    internal class NeuronFunctionsBuilder
    {
        public Func<double, double> Linear { get; } = (s) => s;
        public Func<double, double> Sigmoid { get; } = (s) => 1d / (1 + Math.Exp(-s));
    }
}