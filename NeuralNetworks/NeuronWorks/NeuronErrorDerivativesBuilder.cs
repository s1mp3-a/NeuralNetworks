using System;

namespace NeuralNetworks.NeuronWorks
{
    internal class NeuronErrorDerivativesBuilder
    {
        public Func<double, double, double, double> Sigmoid { get; } =
            (yK, oK, xK) => -(yK - oK) * oK * (1 - oK) * xK;

        public Func<double, double, double, double> Linear { get; } =
            (yK, oK, xK) => -2d * (yK - oK) * xK;
    }
}