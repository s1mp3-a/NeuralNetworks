namespace NeuralNetworks.NeuronWorks
{
    internal static class NeuronFunctions
    {
        private static readonly NeuronFunctionsBuilder _singletonFunctions = new();
        private static readonly NeuronErrorDerivativesBuidler _singletonDerivatives = new();
        
        public static NeuronFunctionsBuilder Activators => _singletonFunctions;
        public static NeuronErrorDerivativesBuidler Derivatives => _singletonDerivatives;
    }
}