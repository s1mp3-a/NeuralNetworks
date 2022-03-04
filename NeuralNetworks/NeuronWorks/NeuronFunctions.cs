namespace NeuralNetworks.NeuronWorks
{
    internal static class NeuronFunctions
    {
        private static readonly NeuronFunctionsBuilder _singleton = new();
        
        public static NeuronFunctionsBuilder Activators => _singleton;
    }
}