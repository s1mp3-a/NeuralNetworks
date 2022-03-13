namespace NeuralNetworks.NeuronWorks
{
    internal static class NeuronFunctions
    {
        public static NeuronFunctionsBuilder Activators { get; } = new();
        public static NeuronErrorDerivativesBuilder Derivatives { get; } = new();

        public static NeuronFunctionsVector VectorFunc { get; } = new();
    }
}