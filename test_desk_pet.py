import unittest
import os
import json
import shutil
import tempfile
from desk_pet import SmallBrainNN, DeskPet, MockWeDo2Hub, train_default_brain

class TestDeskPet(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.initial_state = {
            "pet_name": "TestKepler",
            "profile": "Puppy",
            "level": 1,
            "xp": 0,
            "energy": 80,
            "happiness": 70,
            "hunger": 30,
            "trainer_hp": 100
        }
        self.hub = MockWeDo2Hub("Test Mock Hub")
        self.pet = DeskPet(self.hub, self.initial_state)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_leveling_system(self):
        # Verify initial level
        self.assertEqual(self.pet.level, 1)
        self.assertEqual(self.pet.xp, 0)
        
        # Level 1 XP needed: 250 * (1 ** 1.5) = 250
        xp_needed = self.pet.get_xp_needed()
        self.assertEqual(xp_needed, 250)
        
        # Gain XP (not enough to level up)
        self.pet.gain_xp(100)
        self.assertEqual(self.pet.level, 1)
        self.assertEqual(self.pet.xp, 100)
        
        # Gain more XP to level up
        self.pet.gain_xp(160) # Total 260
        self.assertEqual(self.pet.level, 2)
        # XP leftover: 260 - 250 = 10
        self.assertEqual(self.pet.xp, 10)
        
        # Level 2 XP needed: 250 * (2 ** 1.5) = 707
        self.assertEqual(self.pet.get_xp_needed(), 707)

    def test_neural_network_fallback(self):
        # Verify SmallBrainNN initializes and trains
        brain = train_default_brain()
        self.assertIsNotNone(brain)
        
        # Predict an action (feed input representation)
        # Puppy profile: [1, 0, 0, 0, 0, 0]
        # Feed keyword: 1.0 (feed), 0.0 (pet), etc.
        test_input = [1, 0, 0, 0, 0, 0,  1.0, 0,  0.5, 0.5, 0.5, 1.0]
        prediction = brain.forward(test_input)
        
        # Verify output is a valid probability list (length 7)
        self.assertEqual(len(prediction), 7)
        for val in prediction:
            self.assertTrue(0.0 <= val <= 1.0)
            
        # Verify classification index is between 0 and 6
        predicted_class = prediction.index(max(prediction))
        self.assertTrue(0 <= predicted_class <= 6)

    def test_pet_stats_boundaries(self):
        # Stats should stay bounded between 0 and 100
        self.pet.energy = 95
        self.pet.happiness = 95
        self.pet.hunger = 5
        
        # Interact feed should decrease hunger, increase energy
        self.pet.interact_feed()
        self.assertEqual(self.pet.hunger, 0) # bounded to 0
        self.assertEqual(self.pet.energy, 100) # bounded to 100
        
        # Interact poke should decrease happiness
        self.pet.interact_poke()
        self.assertTrue(self.pet.happiness < 95)

    def test_mock_hub_operations(self):
        # Check led color mapping
        self.hub.set_led("blue")
        self.assertEqual(self.hub.current_led, "blue")
        
        self.hub.stop_motor()
        self.assertEqual(self.hub.motor_speed, 0)

if __name__ == "__main__":
    unittest.main()
