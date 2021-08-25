"""Test file for exciting ASE calculator."""
from parameterized import parameterized
import os
import shutil
import numpy as np
import pytest
import tempfile  # Used to create temporary directories for tests.
import unittest
import ase
from ase.build import bulk
import ase.calculators.exciting
import xml.etree.ElementTree as ET
from ase.calculators.calculator import PropertyNotImplementedError
from ase.units import Bohr, Hartree
import unittest.mock as mock

# @pytest.mark.calculator_lite
# @pytest.mark.calculator('exciting')
# def test_exciting_bulk(factory):
#     """System level test. Ensure that at least the call doesn't fail."""
#     atoms = bulk('Si')
#     atoms.calc = factory.calc()
#     energy = atoms.get_potential_energy()
#     print(energy)


# skips tests which need the exciting calculator if the environment variable 'EXCITINGROOT' is not set
need_exciting = pytest.mark.skipif('EXCITINGROOT' not in os.environ.keys(), reason='exciting is not available')


class Test_pytest_Exciting:

    @pytest.fixture(autouse=True)
    def set_up(self):
        """Code to use for all tests at runtime."""
        self.test_folder_name = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.test_folder_name)

    @pytest.fixture
    def calculator(self):
        return ase.calculators.exciting.Exciting(dir=self.test_folder_name, species_path=self.test_folder_name, exciting_binary='cat')

    @pytest.fixture
    def write_example_output(self):
        with open(self.test_folder_name + '/info.xml', 'w') as file:
            file.write(
                '<?xml version="1.0"?><?xml-stylesheet href="http://xml.exciting-code.org/info.xsl" type="text/xsl"?><info date="2021-05-20" time="10:19:47" versionhash="" title="CO2 molecule">  <groundstate status="finished"><scl> <iter iteration="1" rms="0.13693786788540133" rmslog10="-0.863476438356" deltae="188.25708412518716" deltaelog10="2.27475132771" dforcemax="0.0000000000000000" dforcemaxlog10="0.00000000000" chgdst="0.12834967100038533" chgdstlog10="-0.891605240221" fermidos="0.00000000000"><energies totalEnergy="-188.257084125" fermiEnergy="0.616746586122E-01" sum-of-eigenvalues="-97.9564300535" electronic-kinetic="185.235975399" core-electron-kinetic="0.00000000000" Coulomb="-352.373883716" Coulomb-potential="-255.363517564" nuclear-nuclear="-41.4499436165" electron-nuclear="-366.484362634" Hartree="55.5604225350" Madelung="-224.692124934" xc-potential="-27.8288878880" exchange="-19.6003673613" correlation="-1.51880844690"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.3935503877E-02" valence="16.00000000" interstitial="9.643851966" muffin-tin-total="12.35614803">  <atom species="C" muffin-tin="2.767136653"/>  <atom species="O" muffin-tin="4.794505690"/>  <atom species="O" muffin-tin="4.794505690"/></charges><timing timetot="13.1147814298" timeinit="5.91952395695" timemat="0.147973863874" timefv="0.914317981806E-01" timesv="0.00000000000" timerho="0.113212440861" timepot="6.84178079199" timefor="0.858577899635E-03"/> </iter> <iter iteration="2" rms="4.4316270978184739E-002" rmslog10="-1.35343679074" deltae="1.2995622153846682" deltaelog10="0.113797075797" dforcemax="0.85344050804282656" dforcemaxlog10="-0.688267474599E-01" chgdst="6.2228485852996542E-002" chgdstlog10="-1.20601076616" fermidos="0.00000000000"><energies totalEnergy="-186.957521910" fermiEnergy="0.772305773626E-01" sum-of-eigenvalues="-97.5580142459" electronic-kinetic="185.367790711" core-electron-kinetic="0.00000000000" Coulomb="-351.417637757" Coulomb-potential="-255.376694599" nuclear-nuclear="-41.4499436165" electron-nuclear="-364.558693681" Hartree="54.5909995413" Madelung="-223.729290457" xc-potential="-27.5491103579" exchange="-19.3966890211" correlation="-1.51098584290"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.3903043538E-02" valence="16.00000000" interstitial="9.848651449" muffin-tin-total="12.15134855">  <atom species="C" muffin-tin="2.960661504"/>  <atom species="O" muffin-tin="4.595343523"/>  <atom species="O" muffin-tin="4.595343523"/></charges><timing itertime="3.16049176222" timetot="18.0482827921" timeinit="5.91952395695" timemat="0.315128065879" timefv="0.147284053033" timesv="0.00000000000" timerho="0.226509775035" timepot="11.4379324303" timefor="0.190451089293E-02"/> </iter> <iter iteration="3" rms="2.7910067520827191E-002" rmslog10="-1.55423911285" deltae="5.9698463901440846E-002" deltaelog10="-1.22403684354" dforcemax="0.25832695414673029" dforcemaxlog10="-0.587830276651" chgdst="6.3898592576860820E-003" chgdstlog10="-2.19450870746" fermidos="0.00000000000"><energies totalEnergy="-186.897823446" fermiEnergy="0.796421404335E-01" sum-of-eigenvalues="-97.4896793516" electronic-kinetic="185.518947625" core-electron-kinetic="0.00000000000" Coulomb="-351.498098170" Coulomb-potential="-255.444937332" nuclear-nuclear="-41.4499436165" electron-nuclear="-364.651371774" Hartree="54.6032172206" Madelung="-223.775629503" xc-potential="-27.5636896439" exchange="-19.4074018661" correlation="-1.51127103494"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.3907027817E-02" valence="16.00000000" interstitial="9.838287656" muffin-tin-total="12.16171234">  <atom species="C" muffin-tin="2.934804179"/>  <atom species="O" muffin-tin="4.613454083"/>  <atom species="O" muffin-tin="4.613454083"/></charges><timing itertime="3.24429357983" timetot="23.1430063331" timeinit="5.91952395695" timemat="0.460399522912" timefv="0.202954359120" timesv="0.00000000000" timerho="0.337787029799" timepot="16.2195761134" timefor="0.276535097510E-02"/> </iter> <iter iteration="4" rms="5.3940685824145742E-003" rmslog10="-2.26808353605" deltae="0.22211241363089584" deltaelog10="-0.653427168508" dforcemax="5.7669076883621800E-002" dforcemaxlog10="-1.23905700034" chgdst="5.6493662245052237E-003" chgdstlog10="-2.24800027088" fermidos="0.00000000000"><energies totalEnergy="-186.675711032" fermiEnergy="0.858803585179E-01" sum-of-eigenvalues="-97.3366687241" electronic-kinetic="185.827919656" core-electron-kinetic="0.00000000000" Coulomb="-351.592877011" Coulomb-potential="-255.611277893" nuclear-nuclear="-41.4499436165" electron-nuclear="-364.674588896" Hartree="54.5316555016" Madelung="-223.787238065" xc-potential="-27.5533104870" exchange="-19.4001486701" correlation="-1.51060500706"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.3897997780E-02" valence="16.00000000" interstitial="9.848523639" muffin-tin-total="12.15147636">  <atom species="C" muffin-tin="2.913725713"/>  <atom species="O" muffin-tin="4.618875324"/>  <atom species="O" muffin-tin="4.618875324"/></charges><timing itertime="3.41360818711" timetot="28.1015294653" timeinit="5.91952395695" timemat="0.609567759093" timefv="0.259885576088" timesv="0.00000000000" timerho="0.455699971644" timepot="20.8529694155" timefor="0.388278602622E-02"/> </iter> <iter iteration="5" rms="1.5664801914138501E-003" rmslog10="-2.80507509252" deltae="0.13330648472569351" deltaelog10="-0.875148723714" dforcemax="8.6527928102661703E-002" dforcemaxlog10="-1.06284369527" chgdst="2.3105424273706637E-003" chgdstlog10="-2.63628605233" fermidos="0.00000000000"><energies totalEnergy="-186.542404548" fermiEnergy="0.884354493809E-01" sum-of-eigenvalues="-97.2587004126" electronic-kinetic="185.890817721" core-electron-kinetic="0.00000000000" Coulomb="-351.527231205" Coulomb-potential="-255.602507196" nuclear-nuclear="-41.4499436165" electron-nuclear="-364.552067982" Hartree="54.4747803930" Madelung="-223.725977607" xc-potential="-27.5470109379" exchange="-19.3955681003" correlation="-1.51042296304"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.3897584478E-02" valence="16.00000000" interstitial="9.854367499" muffin-tin-total="12.14563250">  <atom species="C" muffin-tin="2.908689355"/>  <atom species="O" muffin-tin="4.618471573"/>  <atom species="O" muffin-tin="4.618471573"/></charges><timing itertime="3.24940068903" timetot="33.1754948243" timeinit="5.91952395695" timemat="0.795824597124" timefv="0.316849441966" timesv="0.00000000000" timerho="0.572997475741" timepot="25.5656572424" timefor="0.464211008511E-02"/> </iter> <iter iteration="6" rms="2.5065028934889981E-004" rmslog10="-3.60093178970" deltae="4.4702110703610742E-002" deltaelog10="-1.34967197026" dforcemax="7.3245652882352436E-002" dforcemaxlog10="-1.13521814552" chgdst="4.8353079040683311E-004" chgdstlog10="-3.31557586558" fermidos="0.00000000000"><energies totalEnergy="-186.587106658" fermiEnergy="0.876877476440E-01" sum-of-eigenvalues="-97.2841926624" electronic-kinetic="185.868840672" core-electron-kinetic="0.00000000000" Coulomb="-351.547693463" Coulomb-potential="-255.603031263" nuclear-nuclear="-41.4499436165" electron-nuclear="-364.592468430" Hartree="54.4947185831" Madelung="-223.746177831" xc-potential="-27.5500020714" exchange="-19.3977378637" correlation="-1.51051600393"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.3897567590E-02" valence="16.00000000" interstitial="9.851647936" muffin-tin-total="12.14835206">  <atom species="C" muffin-tin="2.909987347"/>  <atom species="O" muffin-tin="4.619182358"/>  <atom species="O" muffin-tin="4.619182358"/></charges><timing itertime="3.27551500499" timetot="38.2220347554" timeinit="5.91952395695" timemat="0.982309825253" timefv="0.409329832997" timesv="0.00000000000" timerho="0.686726670712" timepot="30.2188185165" timefor="0.532595301047E-02"/> </iter> <iter iteration="7" rms="2.7118852063365472E-005" rmslog10="-4.56672869803" deltae="5.9350234765815912E-003" deltaelog10="-2.22657755881" dforcemax="1.5136944158681692E-002" dforcemaxlog10="-1.81996179121" chgdst="1.1242161092008749E-004" chgdstlog10="-3.94915019587" fermidos="0.00000000000"><energies totalEnergy="-186.581171635" fermiEnergy="0.877506374688E-01" sum-of-eigenvalues="-97.2810987637" electronic-kinetic="185.870484526" core-electron-kinetic="0.00000000000" Coulomb="-351.543978429" Coulomb-potential="-255.602343141" nuclear-nuclear="-41.4499436165" electron-nuclear="-364.585726485" Hartree="54.4916916721" Madelung="-223.742806859" xc-potential="-27.5492401491" exchange="-19.3971838968" correlation="-1.51049383463"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.3897795185E-02" valence="16.00000000" interstitial="9.852298146" muffin-tin-total="12.14770185">  <atom species="C" muffin-tin="2.910148946"/>  <atom species="O" muffin-tin="4.618776454"/>  <atom species="O" muffin-tin="4.618776454"/></charges><timing itertime="3.18502099393" timetot="43.2349398122" timeinit="5.91952395695" timemat="1.12796146329" timefv="0.466788080987" timesv="0.00000000000" timerho="0.799096804811" timepot="34.9155304732" timefor="0.603903294541E-02"/> </iter> <iter iteration="8" rms="9.0813354664960076E-006" rmslog10="-5.04185028110" deltae="4.5413804184590845E-004" deltaelog10="-3.34281211697" dforcemax="2.0279226787404819E-003" dforcemaxlog10="-2.69294860794" chgdst="9.1788763918806948E-006" chgdstlog10="-5.03721047857" fermidos="0.00000000000"><energies totalEnergy="-186.580717497" fermiEnergy="0.877643061332E-01" sum-of-eigenvalues="-97.2808737667" electronic-kinetic="185.870838679" core-electron-kinetic="0.00000000000" Coulomb="-351.543896775" Coulomb-potential="-255.602496481" nuclear-nuclear="-41.4499436165" electron-nuclear="-364.585409836" Hartree="54.4914566774" Madelung="-223.742648535" xc-potential="-27.5492159643" exchange="-19.3971664979" correlation="-1.51049290257"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.3897755541E-02" valence="16.00000000" interstitial="9.852316578" muffin-tin-total="12.14768342">  <atom species="C" muffin-tin="2.910140835"/>  <atom species="O" muffin-tin="4.618771293"/>  <atom species="O" muffin-tin="4.618771293"/></charges><timing itertime="3.30073715793" timetot="48.3286040870" timeinit="5.91952395695" timemat="1.30547482520" timefv="0.561847405042" timesv="0.00000000000" timerho="0.909504042706" timepot="39.6252788522" timefor="0.697500491515E-02"/> </iter> <iter iteration="9" rms="1.2771314825234658E-006" rmslog10="-5.89376438919" deltae="1.6094622614559739E-004" deltaelog10="-3.79331920216" dforcemax="3.5682915406024668E-004" dforcemaxlog10="-3.44753966963" chgdst="4.6776891232187917E-006" chgdstlog10="-5.32996864455" fermidos="0.00000000000"><energies totalEnergy="-186.580878443" fermiEnergy="0.877613461235E-01" sum-of-eigenvalues="-97.2809376954" electronic-kinetic="185.870797979" core-electron-kinetic="0.00000000000" Coulomb="-351.543998361" Coulomb-potential="-255.602495030" nuclear-nuclear="-41.4499436165" electron-nuclear="-364.585614458" Hartree="54.4915597142" Madelung="-223.742750846" xc-potential="-27.5492406445" exchange="-19.3971844490" correlation="-1.51049361234"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.3897757016E-02" valence="16.00000000" interstitial="9.852298031" muffin-tin-total="12.14770197">  <atom species="C" muffin-tin="2.910127068"/>  <atom species="O" muffin-tin="4.618787450"/>  <atom species="O" muffin-tin="4.618787450"/></charges><timing itertime="3.31778057106" timetot="53.4814852485" timeinit="5.91952395695" timemat="1.48716914514" timefv="0.656085296068" timesv="0.00000000000" timerho="0.984827227658" timepot="44.4260175030" timefor="0.786211970262E-02"/> </iter> <iter iteration="10" rms="3.1505880344579381E-008" rmslog10="-7.50160838072" deltae="4.3632094843815139E-006" deltaelog10="-5.36019393537" dforcemax="2.9860613584897777E-005" dforcemaxlog10="-4.52490127251" chgdst="5.6946120700438382E-007" chgdstlog10="-6.24453585571" fermidos="0.00000000000"><energies totalEnergy="-186.580874080" fermiEnergy="0.877614112151E-01" sum-of-eigenvalues="-97.2809387790" electronic-kinetic="185.870794824" core-electron-kinetic="0.00000000000" Coulomb="-351.543992211" Coulomb-potential="-255.602494770" nuclear-nuclear="-41.4499436165" electron-nuclear="-364.585602420" Hartree="54.4915538250" Madelung="-223.742744826" xc-potential="-27.5492388338" exchange="-19.3971831286" correlation="-1.51049356440"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.3897756938E-02" valence="16.00000000" interstitial="9.852299119" muffin-tin-total="12.14770088">  <atom species="C" muffin-tin="2.910129042"/>  <atom species="O" muffin-tin="4.618785919"/>  <atom species="O" muffin-tin="4.618785919"/></charges><timing itertime="3.22312420094" timetot="58.4837630973" timeinit="5.91952395695" timemat="1.66824426712" timefv="0.750372393057" timesv="0.00000000000" timerho="1.06288339058" timepot="49.0741754121" timefor="0.856367754750E-02"/> </iter> <iter iteration="11" rms="5.7143670008602962E-009" rmslog10="-8.24303187083" deltae="9.8278127325102105E-007" deltaelog10="-6.00754312753" dforcemax="4.1984657257881919E-006" dforcemaxlog10="-5.37690938783" chgdst="1.6321371828329163E-008" chgdstlog10="-7.78724334114" fermidos="0.00000000000"><energies totalEnergy="-186.580873097" fermiEnergy="0.877614254009E-01" sum-of-eigenvalues="-97.2809382157" electronic-kinetic="185.870795103" core-electron-kinetic="0.00000000000" Coulomb="-351.543991587" Coulomb-potential="-255.602494591" nuclear-nuclear="-41.4499436165" electron-nuclear="-364.585601350" Hartree="54.4915533797" Madelung="-223.742744292" xc-potential="-27.5492387281" exchange="-19.3971830518" correlation="-1.51049356135"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.3897756942E-02" valence="16.00000000" interstitial="9.852299228" muffin-tin-total="12.14770077">  <atom species="C" muffin-tin="2.910129047"/>  <atom species="O" muffin-tin="4.618785862"/>  <atom species="O" muffin-tin="4.618785862"/></charges><timing itertime="3.26755408011" timetot="63.5926985268" timeinit="5.91952395695" timemat="1.84472862026" timefv="0.807147230022" timesv="0.00000000000" timerho="1.17237992375" timepot="53.8393156552" timefor="0.960314064287E-02"/> </iter> <iter iteration="12" rms="4.6355710963320945E-010" rmslog10="-9.33389675371" deltae="2.6177531253779307E-008" deltaelog10="-7.58207131323" dforcemax="4.1203796313782348E-007" dforcemaxlog10="-6.38506276838" chgdst="2.2750894609227182E-009" chgdstlog10="-8.64300152137" fermidos="0.00000000000"><energies totalEnergy="-186.580873123" fermiEnergy="0.877614256420E-01" sum-of-eigenvalues="-97.2809382275" electronic-kinetic="185.870795126" core-electron-kinetic="0.00000000000" Coulomb="-351.543991630" Coulomb-potential="-255.602494618" nuclear-nuclear="-41.4499436165" electron-nuclear="-364.585601410" Hartree="54.4915533959" Madelung="-223.742744321" xc-potential="-27.5492387355" exchange="-19.3971830571" correlation="-1.51049356153"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.3897756942E-02" valence="16.00000000" interstitial="9.852299221" muffin-tin-total="12.14770078">  <atom species="C" muffin-tin="2.910129042"/>  <atom species="O" muffin-tin="4.618785868"/>  <atom species="O" muffin-tin="4.618785868"/></charges><timing itertime="3.36682122084" timetot="68.7488460110" timeinit="5.91952395695" timemat="2.01547148218" timefv="0.862658350030" timesv="0.00000000000" timerho="1.27967146295" timepot="58.6607705201" timefor="0.107502387837E-01"/> </iter> <iter iteration="13" rms="9.4496365554005778E-011" rmslog10="-10.0245848947" deltae="1.2413067906891229E-008" deltaelog10="-7.90612086875" dforcemax="7.8591106532355237E-008" dforcemaxlog10="-7.10462659649" chgdst="1.6306341386840936E-010" chgdstlog10="-9.78764346960" fermidos="0.00000000000"><energies totalEnergy="-186.580873136" fermiEnergy="0.877614254361E-01" sum-of-eigenvalues="-97.2809382335" electronic-kinetic="185.870795122" core-electron-kinetic="0.00000000000" Coulomb="-351.543991638" Coulomb-potential="-255.602494619" nuclear-nuclear="-41.4499436165" electron-nuclear="-364.585601424" Hartree="54.4915534027" Madelung="-223.742744329" xc-potential="-27.5492387368" exchange="-19.3971830581" correlation="-1.51049356157"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.3897756942E-02" valence="16.00000000" interstitial="9.852299219" muffin-tin-total="12.14770078">  <atom species="C" muffin-tin="2.910129042"/>  <atom species="O" muffin-tin="4.618785869"/>  <atom species="O" muffin-tin="4.618785869"/></charges><timing itertime="3.25505611021" timetot="73.7473959872" timeinit="5.91952395695" timemat="2.16056781728" timefv="0.954947651131" timesv="0.00000000000" timerho="1.39292783896" timepot="63.3079160070" timefor="0.115127158351E-01"/> </iter> <iter iteration="13" rms="9.5700833536992441E-012" rmslog10="-11.0190842796" deltae="2.0959589619451435E-009" deltaelog10="-8.67861722492" dforcemax="7.3038805059599810E-008" dforcemaxlog10="-7.13644634063" chgdst="1.3346933466055611E-010" chgdstlog10="-9.87461850447" fermidos="0.00000000000"><energies totalEnergy="-186.678769835" fermiEnergy="0.831408797965E-01" sum-of-eigenvalues="-98.0578551862" electronic-kinetic="186.958679233" core-electron-kinetic="0.00000000000" Coulomb="-352.438117832" Coulomb-potential="-257.082144217" nuclear-nuclear="-33.5493288534" electron-nuclear="-380.695433741" Hartree="61.8066447622" Madelung="-223.897045724" xc-potential="-27.9343902019" exchange="-19.6754841392" correlation="-1.52384709605"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.4018153924E-02" valence="16.00000000" interstitial="9.452679627" muffin-tin-total="12.54732037">  <atom species="C" muffin-tin="3.115544983"/>  <atom species="O" muffin-tin="4.715887695"/>  <atom species="O" muffin-tin="4.715887695"/></charges><timing itertime="283.082942876" timetot="485.344225137" timeinit="6.07864083606" timemat="16.3713530090" timefv="6.39339693938" timesv="0.00000000000" timerho="8.88467625412" timepot="445.271292400" timefor="2.34486569837"/> </iter> <structure forceMax="0.246194007063E-04"><crystal unitCellVolume="250.0000000" BrillouinZoneVolume="0.9922008538" nktot="0" ngridk="1    1    1">  <basevect>10.00000000       0.000000000       0.000000000</basevect>  <basevect>0.000000000       5.000000000       0.000000000</basevect>  <basevect>0.000000000       0.000000000       5.000000000</basevect>  <reciprvect>0.6283185307       0.000000000       0.000000000</reciprvect>  <reciprvect>0.000000000       1.256637061       0.000000000</reciprvect>  <reciprvect>0.000000000       0.000000000       1.256637061</reciprvect></crystal><species chemicalSymbol="C">  <atom x="0.00000000000" y="0.00000000000" z="0.00000000000">    <forces Magnitude="0.00000000000">      <Hellmann-Feynman x="-0.740148683083E-16" y="0.00000000000" z="-0.205432527401E-32"/>      <core-correction x="0.185037170771E-16" y="0.00000000000" z="-0.154074395551E-32"/>      <IBS x="-0.394430452611E-30" y="0.00000000000" z="-0.924446373306E-32"/>      <totalforce x="0.00000000000" y="0.00000000000" z="0.00000000000"/>    </forces>  </atom></species><species chemicalSymbol="O">  <atom x="0.215273901043" y="0.00000000000" z="0.00000000000">    <forces Magnitude="0.246194007063E-04">      <Hellmann-Feynman x="0.561661217052" y="0.00000000000" z="0.410865054803E-32"/>      <core-correction x="-0.417901848779" y="-0.308148791102E-32" z="0.00000000000"/>      <IBS x="-0.143783987673" y="0.00000000000" z="0.462223186653E-32"/>      <totalforce x="-0.246194007063E-04" y="0.00000000000" z="0.00000000000"/>    </forces>  </atom>  <atom x="0.784726098957" y="0.00000000000" z="0.00000000000">    <forces Magnitude="0.246194007063E-04">      <Hellmann-Feynman x="-0.561661217052" y="0.00000000000" z="-0.205432527401E-32"/>      <core-correction x="0.417901848779" y="0.308148791102E-32" z="0.154074395551E-32"/>      <IBS x="0.143783987673" y="0.00000000000" z="0.462223186653E-32"/>      <totalforce x="0.246194007063E-04" y="0.00000000000" z="0.00000000000"/>    </forces>  </atom></species> </structure></scl></groundstate></info>')

    @pytest.fixture
    def write_unfinished_output(self):
        with open(self.test_folder_name + '/info.xml', 'w') as file:
            file.write(
                '<?xml version="1.0"?><?xml-stylesheet href="http://xml.exciting-code.org/info.xsl" type="text/xsl"?><info date="2021-05-20" time="10:19:47" versionhash="" title="CO2 molecule">  <groundstate status="not finished"><scl>  <iter iteration="13" rms="9.5700833536992441E-012" rmslog10="-11.0190842796" deltae="2.0959589619451435E-009" deltaelog10="-8.67861722492" dforcemax="7.3038805059599810E-008" dforcemaxlog10="-7.13644634063" chgdst="1.3346933466055611E-010" chgdstlog10="-9.87461850447" fermidos="0.00000000000"><energies totalEnergy="-186.678769835" fermiEnergy="0.831408797965E-01" sum-of-eigenvalues="-98.0578551862" electronic-kinetic="186.958679233" core-electron-kinetic="0.00000000000" Coulomb="-352.438117832" Coulomb-potential="-257.082144217" nuclear-nuclear="-33.5493288534" electron-nuclear="-380.695433741" Hartree="61.8066447622" Madelung="-223.897045724" xc-potential="-27.9343902019" exchange="-19.6754841392" correlation="-1.52384709605"/><charges totalcharge="22.00000000" core="6.000000000" core_leakage="0.4018153924E-02" valence="16.00000000" interstitial="9.452679627" muffin-tin-total="12.54732037">  <atom species="C" muffin-tin="3.115544983"/>  <atom species="O" muffin-tin="4.715887695"/>  <atom species="O" muffin-tin="4.715887695"/></charges><timing itertime="283.082942876" timetot="485.344225137" timeinit="6.07864083606" timemat="16.3713530090" timefv="6.39339693938" timesv="0.00000000000" timerho="8.88467625412" timepot="445.271292400" timefor="2.34486569837"/>  </iter>  <structure forceMax="0.246194007063E-04"><crystal unitCellVolume="250.0000000" BrillouinZoneVolume="0.9922008538" nktot="0" ngridk="111">  <basevect>10.00000000   0.000000000   0.000000000</basevect>  <basevect>0.000000000   5.000000000   0.000000000</basevect>  <basevect>0.000000000   0.000000000   5.000000000</basevect>  <reciprvect>0.6283185307   0.000000000   0.000000000</reciprvect>  <reciprvect>0.000000000   1.256637061   0.000000000</reciprvect>  <reciprvect>0.000000000   0.000000000   1.256637061</reciprvect></crystal><species chemicalSymbol="C">  <atom x="0.00000000000" y="0.00000000000" z="0.00000000000"><forces Magnitude="0.00000000000">  <Hellmann-Feynman x="-0.740148683083E-16" y="0.00000000000" z="-0.205432527401E-32"/>  <core-correction x="0.185037170771E-16" y="0.00000000000" z="-0.154074395551E-32"/>  <IBS x="-0.394430452611E-30" y="0.00000000000" z="-0.924446373306E-32"/>  <totalforce x="0.00000000000" y="0.00000000000" z="0.00000000000"/></forces>  </atom></species><species chemicalSymbol="O">  <atom x="0.215273901043" y="0.00000000000" z="0.00000000000"><forces Magnitude="0.246194007063E-04">  <Hellmann-Feynman x="0.561661217052" y="0.00000000000" z="0.410865054803E-32"/>  <core-correction x="-0.417901848779" y="-0.308148791102E-32" z="0.00000000000"/>  <IBS x="-0.143783987673" y="0.00000000000" z="0.462223186653E-32"/>  <totalforce x="-0.246194007063E-04" y="0.00000000000" z="0.00000000000"/></forces>  </atom>  <atom x="0.784726098957" y="0.00000000000" z="0.00000000000"><forces Magnitude="0.246194007063E-04">  <Hellmann-Feynman x="-0.561661217052" y="0.00000000000" z="-0.205432527401E-32"/>  <core-correction x="0.417901848779" y="0.308148791102E-32" z="0.154074395551E-32"/>  <IBS x="0.143783987673" y="0.00000000000" z="0.462223186653E-32"/>  <totalforce x="0.246194007063E-04" y="0.00000000000" z="0.00000000000"/></forces>  </atom></species>  </structure></scl>  </groundstate></info>')

    @parameterized.expand([[
            (3, 3, 3), '/fshome/chm/git/exciting/bin/excitingser',
            '3 3 3', '/fshome/chm/git/exciting/bin/excitingser']])
    def test_exciting_constructor(
            self, kpts, exciting_binary: str,
            expected_kpts, expected_exciting_binary: str):
        """Test write an input for exciting."""
        calc_dir = 'ase/test/calculator/exciting'
        exciting_calc = ase.calculators.exciting.Exciting(
            dir=calc_dir,
            kpts=kpts,
            species_path=self.test_folder_name,
            exciting_binary=exciting_binary,
            maxscl=3)
        # groundstate attribute ngridk returns the calculator's kpts
        assert exciting_calc.groundstate_attributes['ngridk'] == expected_kpts
        assert exciting_calc.dir == calc_dir
        assert exciting_calc.species_path == self.test_folder_name
        assert exciting_calc.exciting_binary == expected_exciting_binary
        # Should be set to False at initialization.
        assert not exciting_calc.converged
        # Should be false by default unless arg is passed to constructor.
        assert not exciting_calc.autormt
        # Should be true by default unless arg is passed to constructor.
        assert exciting_calc.tshift

    def test_exciting_constructor_2(self):
        exciting_root = os.environ.pop('EXCITINGROOT', default=None)
        with pytest.raises(RuntimeError, match='No species path given and no EXCITINGROOT '
                    'local var found'):
            ase.calculators.exciting.Exciting()
        if exciting_root is not None:
            os.environ['EXCITINGROOT'] =exciting_root

    def test_exciting_constructor_3(self):
        os.mkdir(self.test_folder_name + '/species')
        exciting_root = os.environ.pop('EXCITINGROOT', default=None)
        os.environ['EXCITINGROOT'] = self.test_folder_name
        calc = ase.calculators.exciting.Exciting()
        assert calc.species_path == self.test_folder_name + '/species'
        if exciting_root is not None:
            os.environ['EXCITINGROOT'] =exciting_root
        else:
            os.environ.pop('EXCITINGROOT', default=None)

    def test_exciting_constructor_4(self):
        with pytest.raises(RuntimeError, match='Species path given'):
            calc = ase.calculators.exciting.Exciting(species_path=self.test_folder_name + '/species')

    def test_exciting_constructor_5(self):
        calc = ase.calculators.exciting.Exciting(ngridk='1 2 3')
        assert calc.groundstate_attributes['ngridk'] == '1 2 3'

    def test_write(self, calculator):
        """Test the write method"""
        calculator.dir = calculator.dir + '/test'
        calculator.paramdict = {'number': '2'}
        calculator.write(bulk('Fe'))

    def test_write_2(self):
        calculator = ase.calculators.exciting.Exciting(dir=self.test_folder_name, species_path=self.test_folder_name, title='Test')
        calculator.write(bulk('Fe'))

    def test_dict_to_xml_1(self, calculator):
        element = ET.Element('root')
        dictionary = {'text()': 'test'}
        calculator.dict_to_xml(dictionary, element)
        assert element.text.__eq__('test')

    def test_dict_to_xml_2(self, calculator):
        element = ET.Element('root')
        dictionary = {'number': '2'}
        calculator.dict_to_xml(dictionary, element)
        assert element.attrib.get('number') == '2'

    def test_dict_to_xml_3(self, calculator):
        element = ET.Element('root')
        dictionary = {'sub': [{'text()': 'test', 'number': '2'}]}
        calculator.dict_to_xml(dictionary, element)
        assert isinstance(element.find('./sub'), ET.Element)
        assert element.find('./sub').text == 'test'
        assert element.find('./sub').attrib.get('number') == '2'

    def test_dict_to_xml_4(self, calculator):
        element = ET.Element('root')
        dictionary = {'sub': {'number': '2'}}
        calculator.dict_to_xml(dictionary, element)
        assert isinstance(element.find('./sub'), ET.Element)
        assert element.find('./sub').attrib.get('number') == '2'

    def test_dict_to_xml_5(self, calculator):
        element = ET.Element('root')
        ET.SubElement(element, 'sub')
        ET.SubElement(element, 'sub')
        dictionary = {'sub': {'number': '2'}}
        calculator.dict_to_xml(dictionary, element)
        sub_elements = element.findall('./sub')
        assert len(sub_elements) == 2
        assert sub_elements[0].attrib.get('number') == '2'
        assert len(sub_elements[1].keys()) == 0

    def test_dict_to_xml_6(self, calculator):
        with pytest.raises(TypeError, match='cannot deal with'):
            element = ET.Element('root')
            dictionary = {'sub': 1}
            calculator.dict_to_xml(dictionary, element)

    def test_init(self, calculator):
        atoms = bulk('Fe')
        calculator.initialize(atoms)
        assert calculator.numbers.all() == atoms.get_atomic_numbers().all()

    def test_get_stress(self, calculator):
        atoms = bulk('Fe')
        with pytest.raises(PropertyNotImplementedError):
            calculator.get_stress(atoms)

    def test_get_energy(self, calculator):
        calculator.update = mock.MagicMock()
        calculator.energy = -186.678769835 * Hartree
        assert calculator.get_potential_energy(bulk('Fe')) == -186.678769835 * Hartree

    def test_get_forces(self, calculator):
        calculator.update = mock.MagicMock()
        calculator.forces = np.array([0.246194007063E-04 * Bohr / Hartree, 0, 0])
        assert calculator.get_forces(bulk('Fe')).all() == calculator.forces.all()

    def test_update(self, calculator):
        with open(calculator.dir + '/INFO.OUT', mode='w') as file:
            file.write('test')
        with open(calculator.dir + '/info.xml', mode='w') as file:
            file.write('test')
        calculator.read = mock.MagicMock()
        atoms = bulk('Fe')
        calculator.update(atoms)
        calculator.converged = True
        assert calculator.read.call_count == 1
        calculator.update(atoms)
        assert calculator.read.call_count == 1
        calculator.positions = [[1.0, 2.0, 5.0]]
        calculator.update(atoms)
        assert calculator.read.call_count == 2

    def test_read_1(self, calculator):
        with pytest.raises(RuntimeError, match='Output file'):
            calculator.read()

    def test_read_2(self, calculator, write_example_output):
        calculator.read()
        assert calculator.energy
        assert calculator.energy - (-186.678769835 * Hartree) < 0.000000001
        # <totalforce x="0.246194007063E-04" y="0.00000000000" z="0.00000000000"/>
        assert calculator.forces.all() == np.array([0.246194007063E-04*Bohr/Hartree, 0, 0]).all()

    def test_read_3(self, calculator, write_unfinished_output):
        with pytest.raises(RuntimeError, match='Calculation did not converge.'):
            calculator.read()
            assert calculator.energy
            assert calculator.energy - (-186.678769835 * Hartree) < 0.000000001
            assert calculator.forces.all() == np.array([0.246194007063E-04 * Bohr / Hartree, 0, 0]).all()

    @pytest.mark.slow
    @need_exciting
    def test_calculate(self):
        calc = ase.calculators.exciting.Exciting(dir=self.test_folder_name)
        atoms = bulk('Fe')
        calc.get_potential_energy(atoms)
        calc.get_forces(atoms)
        assert calc.positions.all() == atoms.get_positions().all()
        assert calc.cell.all() == atoms.get_cell().all()
        assert calc.pbc.all() == atoms.get_pbc().all()
        assert calc.energy


class TestExciting(unittest.TestCase):
    """Test class for all exciting unit tests."""

    def setUp(self):
        """Code to use for all tests at runtime."""
        self.test_folder_name = tempfile.mkdtemp()
        self.nitrous_oxide_atoms_obj = ase.Atoms(
            'N3O',
            positions=[
                (0, 0, 0), (1, 0, 0),
                (0, 0, 1), (0.5, 0.5, 0.5)],
            cell=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            pbc=True)
        self.calc_dir = tempfile.mkdtemp()
        self.exciting_calc_obj = ase.calculators.exciting.Exciting(
            dir=self.calc_dir,
            kpts='3 3 3',
            species_path=self.test_folder_name,
            exciting_binary='/fshome/chm/git/exciting/bin/excitingser',
            maxscl=3)

    def test_add_attributes_to_element_tree(self):
        """Test adding attributes to our element tree."""
        # Default False in Exciting class constructor.
        expected_autormt = 'false'
        expected_tshift = 'true'
        expected_tforce = 'true'
        expected_maxscl = 3
        root = self.exciting_calc_obj.add_attributes_to_element_tree(
            atoms=self.nitrous_oxide_atoms_obj)
        # Let's now check that the new attributes have been added to the
        # element tree.
        self.assertEqual(
            root.find('structure').attrib['speciespath'],
            self.test_folder_name)
        self.assertEqual(
            root.find('structure').attrib['autormt'],
            expected_autormt)
        print(root.find('structure').attrib['tshift'])
        self.assertEqual(
            root.find('structure').attrib['tshift'],
            expected_tshift)
        self.assertEqual(
            root.find('groundstate').attrib['tforce'],
            expected_tforce)
        self.assertEqual(
            root.find('groundstate').attrib['maxscl'],
            str(expected_maxscl))
        self.assertEqual(
            root.find('groundstate').attrib['ngridk'],
            '3   3   3')

    def test_add_attributes_to_element_tree_with_param_dict(self):
        """Test adding attributes to element tree via param dict."""
        expected_maxscl = 3
        self.test_folder_name = tempfile.mkdtemp()
        self.nitrous_oxide_atoms_obj = ase.Atoms(
            'N3O',
            positions=[
                (0, 0, 0), (1, 0, 0),
                (0, 0, 1), (0.5, 0.5, 0.5)],
            cell=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            pbc=True)
        param_dict = {'maxscl': '3'}
        exciting_calc_obj = ase.calculators.exciting.Exciting(
            dir=self.calc_dir,
            kpts='3 3 3',
            species_path=self.test_folder_name,
            exciting_binary='/fshome/chm/git/exciting/bin/excitingser',
            param_dict=param_dict)

        root = exciting_calc_obj.add_attributes_to_element_tree(
            atoms=self.nitrous_oxide_atoms_obj)
        self.assertEqual(
            root.find('structure').attrib['speciespath'],
            self.test_folder_name)
        # Right now the maxscl param is passed in through a param dict to the
        # exciting calculator object.
        self.assertEqual(
            root.attrib['maxscl'],
            str(expected_maxscl))

    def test_dict_to_xml(self):
        """Test adding a parameter dict to an element tree."""
        fake_xml_string = """<?xml version = "1.0"?>
        <data>
            <country_name> = "Liechtenstein">
                <rank>  1 </rank>
                </country_name>
        </data>"""
        root = ET.fromstring(fake_xml_string)
        self.assertFalse(hasattr(root, 'goose_egg'))
        fake_dict = {'goose_egg': 'true'}
        self.exciting_calc_obj.dict_to_xml(pdict=fake_dict, element=root)
        self.assertEqual(root.attrib['goose_egg'], 'true')

    def test_write(self):
        """Test write ase atoms file into an exciting input file."""
        self.exciting_calc_obj.write(atoms=self.nitrous_oxide_atoms_obj)
        self.assertTrue(os.path.isfile(self.calc_dir + '/input.xml'))
        # Now read the xml file and make sure it has the right chemical
        # symbols in the right place (e.g. under structure/species).
        tree = ET.parse(self.calc_dir + '/input.xml')
        root = tree.getroot()
        # assert the chemical symbols are N and O
        species = root.findall("structure/species")
        # We want a list of the symbols for each species we have
        species = [species[i].attrib for i in range(len(species))]
        symbols = [attribute['chemicalSymbol'] for attribute in species]
        expected_chemical_symbols = ['N', 'O']
        self.assertListEqual(symbols, expected_chemical_symbols)

    def test_read(self):
        """Test reading the exciting output info.xml file."""

        xml_string = """<?xml version="1.0"?>
            <?xml-stylesheet href="http://xml.exciting-code.org/info.xsl" type="text/xsl"?>
            <info date="2016-09-01" time="17:52:17" versionhash="" title="AFM Fe-bcc">
            <groundstate status="finished">
                <scl>
                <iter iteration="20" rms="1.6141002835615733E-008" rmslog10="-7.79206948619" deltae="8.0951858763000928E-006" deltaelog10="-5.09177317481" chgdst="2.0403852911831260E-008" chgdstlog10="-7.69028781589" fermidos="132.482182013">
                    <energies totalEnergy="-2543.25348773" fermiEnergy="0.426565746417" sum-of-eigenvalues="-1466.47893261" electronic-kinetic="2577.53462387" core-electron-kinetic="0.00000000000" Coulomb="-5011.26002189" Coulomb-potential="-3900.52860564" nuclear-nuclear="-382.039729813" electron-nuclear="-5357.91197852" Hartree="728.691686443" Madelung="-3060.99571907" xc-potential="-143.339250380" xc-effective-B-field="-0.120729244350" external-B-field="0.00000000000" exchange="-106.757424516" correlation="-2.77066519350"/>
                    <charges totalcharge="52.00000000" core="20.00000000" core_leakage="0.3814761840E-09" valence="32.00000000" interstitial="3.627763643" muffin-tin-total="48.37223636">
                    <atom species="Fe" muffin-tin="24.18611818"/>
                    <atom species="Fe" muffin-tin="24.18611818"/>
                    </charges>
                    <timing itertime="48.5351307980" timetot="1230.56626552" timeinit="0.834124858000" timemat="42.4505247870" timefv="19.9505684140" timesv="814.310320974" timerho="332.526011722" timepot="20.4947147610" timefor="0.00000000000"/>
                    <moments>
                    <momtot x="0.451215738635E-16"/>
                    <interstitial x="0.451215738635E-16"/>
                    <mommttot x="0.00000000000"/>
                    <atom species="Fe">
                        <mommt x="-1.70899579576"/>
                    </atom>
                    <atom species="Fe">
                        <mommt x="1.70899579576"/>
                    </atom>
                    </moments>
                </iter>
                <structure>
                    <crystal unitCellVolume="140.6080000" BrillouinZoneVolume="1.764125892" nktot="0" ngridk="6    6    6">
                    <basevect>5.200000000       0.000000000       0.000000000</basevect>
                    <basevect>0.000000000       5.200000000       0.000000000</basevect>
                    <basevect>0.000000000       0.000000000       5.200000000</basevect>
                    <reciprvect>1.208304867       0.000000000       0.000000000</reciprvect>
                    <reciprvect>0.000000000       1.208304867       0.000000000</reciprvect>
                    <reciprvect>0.000000000       0.000000000       1.208304867</reciprvect>
                    </crystal>
                    <species chemicalSymbol="Fe">
                    <atom x="0.00000000000" y="0.00000000000" z="0.00000000000"/>
                    <atom x="0.500010000000" y="0.500010000000" z="0.500010000000"/>
                    </species>
                </structure>
                </scl>
            </groundstate>
            </info>"""
        root = ET.fromstring(xml_string)
        pretty_string = ase.calculators.exciting.prettify(root)
        test_xml_file = os.path.join(self.calc_dir, 'info.xml')
        with open(test_xml_file, 'w') as f:
            f.write(pretty_string)
        # Now read the file, it should find the info.xml file automatically.
        self.exciting_calc_obj.read()
        expected_energy = -2543.25348773*Hartree
        self.assertAlmostEqual(self.exciting_calc_obj.energy, expected_energy)
