using System;

namespace NeuralNetworks.NeuronWorks
{
    internal class NeuronErrorDerivativesBuilder
    {
        public Func<double, double> Sigmoid { get; } =
            x => x * (1 - x);
           /* x => NeuronFunctions.Functions.Sigmoid.Activator(x)
                 * (1 - NeuronFunctions.Functions.Sigmoid.Activator(x));*/

        public Func<double, double> Linear { get; } =
            x => -2d * x;
    }
}