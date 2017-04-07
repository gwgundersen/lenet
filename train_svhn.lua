-- ============================================================================
-- Train LeNet5 classifier on MNIST.
-- ============================================================================

require 'svhn'
require 'lenet'
require 'Trainer'


-------------------------------------------------------------------------------
-- Parameters and constants.

local N_EPOCHS   = 2
local BATCH_SIZE = 20
local N_TRAINING = 73257  -- Actual: 73257
local N_TESTING  = 26032  -- Actual: 26032
local CLASSES    = {'1','2','3','4','5','6','7','8','9','10'}
local GEOMETRY   = {3, 32, 32}
local model      = lenet.new(GEOMETRY)
local criterion  = nn.CrossEntropyCriterion()

-- Create training set and normalize
local trainData = svhn.loadTrainSet(N_TRAINING)
local testData = svhn.loadTestSet(N_TESTING)

local trainer = Trainer(model, criterion, CLASSES, BATCH_SIZE, GEOMETRY)
trainer:train(trainData, N_EPOCHS)
local confusion = trainer:test(testData)
print(confusion)
