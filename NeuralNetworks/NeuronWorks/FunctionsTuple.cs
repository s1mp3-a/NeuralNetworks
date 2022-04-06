using System;

namespace NeuralNetworks.NeuronWorks
{
    internal class FunctionsTuple : Tuple<NeuronFunctionsBuilder, NeuronErrorDerivativesBuilder>
    {
        public FunctionsTuple(NeuronFunctionsBuilder a, NeuronErrorDerivativesBuilder b)
            : base(a, b) { }
        public FunctionTypeTuple Sigmoid => new(Item1.Sigmoid, Item2.Sigmoid);
        public FunctionTypeTuple Linear => new(Item1.Linear, Item2.Linear);
    }

}