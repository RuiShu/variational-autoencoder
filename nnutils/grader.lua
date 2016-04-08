-- Highly stripped down version of torch-totem for gradient checking
-- and speed testing
require 'nn'
grader = {}

local function computeNumGradParams(str)
   print(str)
end

function grader.checkModuleGradInput(mod, input)
   mod:zeroGradParameters()
   local out = mod:forward(input)
   local gradOut = out.new():resizeAs(out):copy(torch.randn(out:nElement()))
   local gradInput = mod:backward(input, gradOut)
   local flatInput = input:storage()
   local flatGradInput = gradInput:storage()
   local h = 1e-4
   local maxDiff = 0
   for i=1,input:nElement() do
      local origVal = flatInput[i]
      flatInput[i] = origVal + h
      local fph = mod:forward(input):cmul(gradOut):sum()
      flatInput[i] = origVal - h
      local fmh = mod:forward(input):cmul(gradOut):sum()
      flatInput[i] = origVal
      local approx = (fph - fmh)/(2*h)
      local diff = torch.abs(approx - flatGradInput[i])
      if diff > maxDiff then
         maxDiff = diff
      end
   end
   print("Max deviation: "..maxDiff)
end

function grader.checkCriterionGradInput(mod, input, target)
   mod:forward(input, target)
   local gradInput, gradTarget = mod:backward(input, target)
   local function check(buffer, gradBuffer)
      local flatBuffer = buffer:storage()
      local flatGradBuffer = gradBuffer:storage()
      local h = 1e-4
      local maxDiff = 0
      for i=1,buffer:nElement() do
         local origVal = flatBuffer[i]
         flatBuffer[i] = origVal + h
         local fph = mod:forward(input, target)
         flatBuffer[i] = origVal - h
         local fmh = mod:forward(input, target)
         flatBuffer[i] = origVal
         local approx = (fph - fmh)/(2*h)
         local diff = torch.abs(approx - flatGradBuffer[i])
         if diff > maxDiff then
            maxDiff = diff
         end
      end
      return maxDiff
   end
   if gradInput then
      print("Max input deviation:",check(input, gradInput))
   end
   if gradTarget then
      print("Max target deviation:",check(target, gradTarget))
   end
end

function grader.speedTest(mod, input, target, nRepeat, modify)
   local nRepeat = nRepeat or 1000
   local cumTime = 0
   for i=1,nRepeat do
      if modify then
         input:copy(torch.randn(input:nElement()))
         target:copy(torch.randn(target:nElement()))
      end
      -->
      t = sys.clock()
      sys.tic()
      -->>>>>>>>>> 
      mod:forward(input, target)
      mod:backward(input, target)
      -->>>>>>>>>>>
      t = sys.toc()
      -->
      cumTime = cumTime + t
   end
   print("Average time: "..cumTime/nRepeat)
end
