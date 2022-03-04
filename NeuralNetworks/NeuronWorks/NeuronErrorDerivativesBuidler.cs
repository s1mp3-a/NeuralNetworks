using System;

namespace NeuralNetworks.NeuronWorks
{
    internal class NeuronErrorDerivativesBuidler
    {
        public Func<double, double, double, double> Sigmoid =>
            (yK, oK, xK) => -(yK - oK) * oK * (1 - oK) * xK;

        public Func<double, double, double, double> Linear =>
            (yK, oK, xK) => -2d * (yK - oK) * xK;
    }
}