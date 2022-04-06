namespace NeuralNetworks.NeuronWorks
{
    internal static class NeuronFunctions
    {
        public static FunctionsTuple Functions { get; } = new(new NeuronFunctionsBuilder(), new NeuronErrorDerivativesBuilder());
    }
}