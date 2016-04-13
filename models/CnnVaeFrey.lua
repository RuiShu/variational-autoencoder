require 'models.CnnVae'
local CnnVaeMnist, parent = torch.class('CnnVaeMnist', 'CnnVae')
local c = require 'trepl.colorize'
require 'nngraph'
require 'nnutils.init'

function CnnVae:build(struct)
   local encoder = nn.Sequential()
   -- conv
   encoder:add(nn.View(1,28,20))
   encoder:add(nn.SpatialConvolution(  1, 50, 2,2, 2,2)):add(nn.ReLU(true)) -- 14 x 10
   encoder:add(nn.SpatialConvolution( 50,100, 2,2, 2,2)):add(nn.ReLU(true)) -- 7 x 5
   encoder:add(nn.SpatialConvolution(100,200, 3,3, 2,2)):add(nn.ReLU(true)) -- 3 x 2
   encoder:add(nn.SpatialConvolution(200,400, 2,3, 2,2)):add(nn.ReLU(true)) -- 1 x 1
   -- linear
   encoder:add(nn.View(400))
   encoder:add(nn.Linear(400,2*struct.z))
   encoder:add(nn.View(2,struct.z))
   local decoder = nn.Sequential()
   -- linear
   decoder:add(nn.Linear(struct.z, struct.h)):add(nn.ReLU(true))
   decoder:add(nn.View(400,1,1))
   -- conv
   decoder:add(nn.SpatialFullConvolution(400,200, 2,3, 2,2)):add(nn.ReLU(true))
   decoder:add(nn.SpatialFullConvolution(200,100, 3,3, 2,2)):add(nn.ReLU(true))
   decoder:add(nn.SpatialFullConvolution(100, 50, 2,2, 2,2)):add(nn.ReLU(true))
   decoder:add(nn.SpatialFullConvolution( 50,  1, 2,2, 2,2))
   decoder:add(nn.View(560))
   decoder:add(nn.Sigmoid())
   -- combine the two
   local input = nn.Identity()()
   local mulv = encoder(input)
   local code = nn.Sampler()(mulv)
   local recon = decoder(code)
   local model = nn.gModule({input},{mulv, recon})
   return encoder, decoder, model
end
