using System;

namespace NeuralNetworks.NeuronWorks
{
    using Act = Func<double, double>;
    using Der = Func<double, double>;
    
    public class FunctionTypeTuple : Tuple<Act, Der>
    {
        public FunctionTypeTuple(Act a, Der b)
            : base(a, b) { }

        public Act Activator => Item1;
        public Der Derivative => Item1;
    }
}